import binascii
from pathlib import Path
import re
import struct
import subprocess
from datetime import datetime
import zlib

import pyvisa


timestamp = datetime.now().strftime("%y%m%d%H%M%S")


def build_output_path(file_extension):
    base_name = f"SCOPE_{timestamp}"
    candidate = Path(f"{base_name}.{file_extension}")

    if not candidate.exists():
        return str(candidate)

    suffix = 1
    while True:
        candidate = Path(f"{base_name}_{suffix}.{file_extension}")
        if not candidate.exists():
            return str(candidate)
        suffix += 1


def get_tektronix_usb_candidates_from_pnputil():
    try:
        output = subprocess.check_output(
            ["pnputil", "/enum-devices", "/connected"],
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return []

    candidates = []
    current_instance_id = None

    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Instance ID:"):
            current_instance_id = stripped.split(":", 1)[1].strip()
            continue

        if not current_instance_id or not stripped.startswith("Device Description:"):
            continue

        device_description = stripped.split(":", 1)[1].strip()
        match = re.match(r"USB\\VID_([0-9A-Fa-f]{4})&PID_([0-9A-Fa-f]{4})\\(.+)", current_instance_id)

        if not match:
            continue

        vid, pid, serial = match.groups()
        if vid.lower() != "0699":
            continue

        if "USB Test and Measurement Device" not in device_description:
            continue

        candidates.append(f"USB0::0x{vid.upper()}::0x{pid.upper()}::{serial}::INSTR")

    return candidates


def find_single_usb_scope(resource_manager):
    usb_resources = []
    seen_resources = set()

    for resource in resource_manager.list_resources():
        if not resource.startswith("USB") or not resource.endswith("::INSTR"):
            continue

        if resource.startswith("USB::"):
            resource = resource.replace("USB::", "USB0::", 1)

        if resource in seen_resources:
            continue

        seen_resources.add(resource)
        usb_resources.append(resource)

    if not usb_resources:
        for resource in get_tektronix_usb_candidates_from_pnputil():
            if resource in seen_resources:
                continue
            seen_resources.add(resource)
            usb_resources.append(resource)

    if not usb_resources:
        raise RuntimeError("No USB oscilloscope resource found.")

    if len(usb_resources) > 1:
        raise RuntimeError(f"Multiple USB instruments detected. Please choose manually: {usb_resources}")

    return usb_resources[0]


def get_instrument_idn(scope):
    return scope.query("*IDN?").strip()


def read_bmp_hardcopy(scope):
    scope.timeout = 20000
    scope.write("HARDCOPY START")
    data = scope.read_raw()
    return extract_bmp_payload(data)


def extract_bmp_payload(data):
    if data.startswith(b"BM"):
        return data

    # Many VISA instruments wrap binary payloads in an IEEE 488.2 block:
    # b"#" + <digits-count> + <payload-length> + <payload> [+ terminator]
    if data.startswith(b"#") and len(data) >= 2:
        digits_count = data[1] - ord("0")
        if 0 <= digits_count <= 9 and len(data) >= 2 + digits_count:
            payload_len_start = 2
            payload_len_end = payload_len_start + digits_count
            payload_len_raw = data[payload_len_start:payload_len_end]

            if payload_len_raw.isdigit():
                payload_len = int(payload_len_raw.decode("ascii"))
                payload_start = payload_len_end
                payload_end = payload_start + payload_len
                payload = data[payload_start:payload_end]
                if payload.startswith(b"BM"):
                    return payload

    bm_offset = data.find(b"BM")
    if bm_offset != -1:
        return data[bm_offset:]

    raise RuntimeError("HARDCOPY did not return a BMP file.")


def read_rigol_png_hardcopy(scope):
    scope.timeout = 20000
    scope.write(":DISPlay:DATA? ON,OFF,PNG")
    data = scope.read_raw()
    return extract_png_payload(data)


def extract_png_payload(data):
    png_signature = b"\x89PNG\r\n\x1a\n"

    if data.startswith(png_signature):
        return data

    # Many instruments wrap binary payloads in an IEEE 488.2 block:
    # b"#" + <digits-count> + <payload-length> + <payload> [+ terminator]
    if data.startswith(b"#") and len(data) >= 2:
        digits_count = data[1] - ord("0")
        if 0 <= digits_count <= 9 and len(data) >= 2 + digits_count:
            payload_len_start = 2
            payload_len_end = payload_len_start + digits_count
            payload_len_raw = data[payload_len_start:payload_len_end]

            if payload_len_raw.isdigit():
                payload_len = int(payload_len_raw.decode("ascii"))
                payload_start = payload_len_end
                payload_end = payload_start + payload_len
                payload = data[payload_start:payload_end]
                if payload.startswith(png_signature):
                    return payload

    png_offset = data.find(png_signature)
    if png_offset != -1:
        return data[png_offset:]

    raise RuntimeError("HARDCOPY did not return a PNG file.")


def capture_scope_image(scope):
    idn = get_instrument_idn(scope)
    idn_upper = idn.upper()

    if "RIGOL" in idn_upper:
        png_data = read_rigol_png_hardcopy(scope)
        return png_data, "png", idn

    # Keep the original Tektronix hardcopy flow as the default path.
    bmp_data = read_bmp_hardcopy(scope)
    png_data = bmp_to_png_bytes(bmp_data)
    return png_data, "png", idn


def make_png_chunk(chunk_type, chunk_data):
    crc = binascii.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(chunk_data))
        + chunk_type
        + chunk_data
        + struct.pack(">I", crc)
    )


def bmp_to_png_bytes(bmp_data):
    if bmp_data[:2] != b"BM":
        raise RuntimeError("Input data is not a BMP file.")

    pixel_offset = struct.unpack_from("<I", bmp_data, 10)[0]
    dib_header_size = struct.unpack_from("<I", bmp_data, 14)[0]
    width = struct.unpack_from("<i", bmp_data, 18)[0]
    height = struct.unpack_from("<i", bmp_data, 22)[0]
    planes = struct.unpack_from("<H", bmp_data, 26)[0]
    bits_per_pixel = struct.unpack_from("<H", bmp_data, 28)[0]
    compression = struct.unpack_from("<I", bmp_data, 30)[0]

    if dib_header_size < 40:
        raise RuntimeError("Unsupported BMP header.")

    if planes != 1 or compression != 0:
        raise RuntimeError("Only uncompressed BMP files are supported.")

    if bits_per_pixel not in (24, 32):
        raise RuntimeError(f"Unsupported BMP pixel format: {bits_per_pixel} bpp")

    top_down = height < 0
    width = abs(width)
    height = abs(height)

    bytes_per_pixel = bits_per_pixel // 8
    row_stride = ((width * bytes_per_pixel + 3) // 4) * 4
    pixel_data = bmp_data[pixel_offset:]

    raw_rows = bytearray()
    row_indexes = range(height) if top_down else range(height - 1, -1, -1)

    for row_index in row_indexes:
        row_start = row_index * row_stride
        row_end = row_start + (width * bytes_per_pixel)
        row = pixel_data[row_start:row_end]

        if len(row) != width * bytes_per_pixel:
            raise RuntimeError("BMP pixel data is truncated.")

        raw_rows.append(0)

        if bits_per_pixel == 24:
            for pixel_index in range(0, len(row), 3):
                blue = row[pixel_index]
                green = row[pixel_index + 1]
                red = row[pixel_index + 2]
                raw_rows.extend((red, green, blue))
        else:
            for pixel_index in range(0, len(row), 4):
                blue = row[pixel_index]
                green = row[pixel_index + 1]
                red = row[pixel_index + 2]
                alpha = row[pixel_index + 3]
                raw_rows.extend((red, green, blue, alpha))

    color_type = 2 if bits_per_pixel == 24 else 6
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    idat = zlib.compress(bytes(raw_rows), level=9)

    return (
        b"\x89PNG\r\n\x1a\n"
        + make_png_chunk(b"IHDR", ihdr)
        + make_png_chunk(b"IDAT", idat)
        + make_png_chunk(b"IEND", b"")
    )


def main():
    rm = pyvisa.ResourceManager()
    try:
        resource_name = find_single_usb_scope(rm)

        scope = rm.open_resource(resource_name)
        scope.timeout = 20000
        scope.chunk_size = 1024000

        try:
            image_data, image_format, instrument_idn = capture_scope_image(scope)
        finally:
            scope.close()

        output_path = build_output_path()

        with open(output_path, "wb") as f:
            f.write(image_data)

        print(f"Connected: {resource_name}")
        print(f"Instrument: {instrument_idn}")
        print(f"Captured: {image_format.upper()} from oscilloscope")
        if "TEKTRONIX" in instrument_idn.upper():
            print("Converted: BMP -> PNG with lossless zlib compression")
        print(f"Saved: {output_path}")
        print("Done")
    finally:
        rm.close()


if __name__ == "__main__":
    main()
