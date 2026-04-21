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


def read_hardcopy(scope):
    scope.timeout = 20000
    scope.write("HARDCOPY START")
    return scope.read_raw()


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


rm = pyvisa.ResourceManager()
resource_name = find_single_usb_scope(rm)

scope = rm.open_resource(resource_name)
scope.timeout = 20000
scope.chunk_size = 1024000

try:
    image_data = read_hardcopy(scope)
finally:
    scope.close()

if image_data.startswith(b"BM"):
    output_data = bmp_to_png_bytes(image_data)
    output_extension = "png"
    capture_format = "BMP"
    saved_format = "PNG"
elif image_data.startswith(b"\x89PNG\r\n\x1a\n"):
    output_data = image_data
    output_extension = "png"
    capture_format = "PNG"
    saved_format = "PNG"
elif image_data.startswith(b"\xff\xd8\xff"):
    output_data = image_data
    output_extension = "jpg"
    capture_format = "JPEG"
    saved_format = "JPEG"
else:
    raise RuntimeError(f"Unsupported hardcopy format. File starts with: {image_data[:16]!r}")

output_path = build_output_path(output_extension)

with open(output_path, "wb") as f:
    f.write(output_data)

print(f"Connected: {resource_name}")
print(f"Captured: {capture_format} from oscilloscope")
if capture_format == "BMP" and saved_format == "PNG":
    print("Converted: BMP -> PNG with lossless zlib compression")
print(f"Saved: {output_path}")
print("Done")
