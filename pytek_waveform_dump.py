from array import array
from datetime import datetime
from pathlib import Path
import time

import h5py
import numpy as np
import pyvisa


HDF5_COMPRESSION = "gzip"
HDF5_COMPRESSION_OPTS = 9
HDF5_FORMAT_VERSION = 2


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
        raise RuntimeError("No USB oscilloscope resource found.")

    if len(usb_resources) > 1:
        raise RuntimeError(f"Multiple USB instruments detected. Please choose manually: {usb_resources}")

    return usb_resources[0]


def get_instrument_idn(scope):
    return scope.query("*IDN?").strip()


def build_hdf5_path():
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    base_name = f"SCOPE_{timestamp}"
    candidate = Path(f"{base_name}.h5")

    if not candidate.exists():
        return candidate

    suffix = 1
    while True:
        candidate = Path(f"{base_name}_{suffix}.h5")
        if not candidate.exists():
            return candidate
        suffix += 1


def get_displayed_channels(scope):
    idn = get_instrument_idn(scope).upper()

    if "RIGOL" in idn:
        return get_displayed_channels_rigol(scope)

    return get_displayed_channels_tek(scope)


def get_displayed_channels_tek(scope):
    displayed_channels = []

    for channel in range(1, 5):
        try:
            is_displayed = scope.query(f"SELECT:CH{channel}?").strip()
        except pyvisa.VisaIOError:
            continue

        if is_displayed == "1":
            displayed_channels.append(channel)

    if not displayed_channels:
        raise RuntimeError("No displayed channels found on the oscilloscope.")

    return displayed_channels


def get_displayed_channels_rigol(scope):
    displayed_channels = []

    for channel in range(1, 5):
        try:
            is_displayed = scope.query(f":CHANnel{channel}:DISPlay?").strip()
        except pyvisa.VisaIOError:
            continue

        if is_displayed == "1":
            displayed_channels.append(channel)

    if not displayed_channels:
        raise RuntimeError("No displayed channels found on the oscilloscope.")

    return displayed_channels


def configure_waveform_transfer_tek(scope, channel):
    scope.write("HEADER 0")
    scope.write(f"DATA:SOURCE CH{channel}")
    scope.write("DATA:START 1")
    scope.write("DATA:STOP 5000000")
    scope.write("DATA:ENCDG RIBINARY")
    scope.write("DATA:WIDTH 1")


def configure_waveform_transfer_rigol(scope, channel):
    scope.write(":STOP")
    time.sleep(0.5)
    scope.write(f":WAV:SOUR CHAN{channel}")
    scope.write(":WAV:MODE RAW")
    scope.write(":WAV:FORM BYTE")


def read_waveform(scope, channel):
    idn = get_instrument_idn(scope).upper()

    if "RIGOL" in idn:
        return read_waveform_rigol(scope, channel)

    return read_waveform_tek(scope, channel)


def read_waveform_tek(scope, channel):
    configure_waveform_transfer_tek(scope, channel)

    y_mult = float(scope.query("WFMOUTPRE:YMULT?").strip())
    y_off = float(scope.query("WFMOUTPRE:YOFF?").strip())
    y_zero = float(scope.query("WFMOUTPRE:YZERO?").strip())
    x_incr = float(scope.query("WFMOUTPRE:XINCR?").strip())
    x_zero = float(scope.query("WFMOUTPRE:XZERO?").strip())
    point_count = int(float(scope.query("WFMOUTPRE:NR_PT?").strip()))

    sample_values = scope.query_binary_values(
        "CURVE?",
        datatype="b",
        is_big_endian=True,
        container=list,
    )
    sample_values = array("b", sample_values)

    tail_artifact_removed = False
    if len(sample_values) >= 2 and sample_values[-1] == 0 and sample_values.count(0) == 1:
        sample_values = sample_values[:-1]
        tail_artifact_removed = True

    if len(sample_values) != point_count:
        point_count = len(sample_values)

    raw_codes = np.frombuffer(sample_values.tobytes(), dtype=np.int8).copy()

    raw_min = int(raw_codes.min())
    raw_max = int(raw_codes.max())
    raw_avg = float(raw_codes.mean())

    voltage_min = (raw_min - y_off) * y_mult + y_zero
    voltage_max = (raw_max - y_off) * y_mult + y_zero
    voltage_avg = (raw_avg - y_off) * y_mult + y_zero

    metadata = {
        "channel": f"CH{channel}",
        "points": point_count,
        "reported_points": point_count,
        "captured_points": len(raw_codes),
        "x_increment_s": x_incr,
        "x_zero_s": x_zero,
        "y_multiplier_v_per_code": y_mult,
        "y_offset_code": y_off,
        "y_zero_v": y_zero,
        "raw_min_code": raw_min,
        "raw_max_code": raw_max,
        "raw_avg_code": raw_avg,
        "voltage_min_v": voltage_min,
        "voltage_max_v": voltage_max,
        "voltage_avg_v": voltage_avg,
        "tail_artifact_removed": tail_artifact_removed,
    }

    return raw_codes, metadata


def read_waveform_rigol(scope, channel):
    configure_waveform_transfer_rigol(scope, channel)

    preamble = scope.query(":WAVeform:PREamble?").strip()
    preamble_parts = [part.strip() for part in preamble.split(",")]

    if len(preamble_parts) < 10:
        raise RuntimeError(f"Unexpected RIGOL preamble: {preamble}")

    point_count = int(float(preamble_parts[2]))
    x_increment = float(preamble_parts[4])
    x_origin = float(preamble_parts[5])
    x_reference = float(preamble_parts[6])
    y_increment = float(preamble_parts[7])
    y_origin = float(preamble_parts[8])
    y_reference = float(preamble_parts[9])

    max_points_per_read = 500000
    transfer_start = time.perf_counter()
    raw_bytes = bytearray()
    start = 1

    while start <= point_count:
        stop = min(start + max_points_per_read - 1, point_count)
        scope.write(f":WAV:STAR {start}")
        scope.write(f":WAV:STOP {stop}")
        chunk = scope.query_binary_values(
            ":WAV:DATA?",
            datatype="B",
            is_big_endian=False,
            container=bytes,
        )
        if not chunk:
            break
        raw_bytes.extend(chunk)
        start = stop + 1

    if raw_bytes:
        raw_codes = np.frombuffer(raw_bytes, dtype=np.uint8).copy()
    else:
        raw_codes = np.empty(0, dtype=np.uint8)

    if len(raw_codes) != point_count:
        point_count = len(raw_codes)

    if point_count == 0:
        raise RuntimeError("RIGOL returned zero waveform points in RAW mode.")

    raw_min = int(raw_codes.min())
    raw_max = int(raw_codes.max())
    raw_avg = float(raw_codes.mean())

    voltage_min = (raw_min - y_origin - y_reference) * y_increment
    voltage_max = (raw_max - y_origin - y_reference) * y_increment
    voltage_avg = (raw_avg - y_origin - y_reference) * y_increment

    metadata = {
        "channel": f"CH{channel}",
        "points": point_count,
        "reported_points": int(float(preamble_parts[2])),
        "captured_points": len(raw_codes),
        "x_increment_s": x_increment,
        "x_zero_s": x_origin,
        "x_reference_index": x_reference,
        "y_increment_v_per_code": y_increment,
        "y_origin_code": y_origin,
        "y_reference_code": y_reference,
        "raw_min_code": raw_min,
        "raw_max_code": raw_max,
        "raw_avg_code": raw_avg,
        "voltage_min_v": voltage_min,
        "voltage_max_v": voltage_max,
        "voltage_avg_v": voltage_avg,
        "tail_artifact_removed": False,
        "transfer_seconds": time.perf_counter() - transfer_start,
        "points_per_read": max_points_per_read,
    }

    return raw_codes, metadata


def build_time_axis(metadata):
    if "x_reference_index" in metadata:
        return (
            (np.arange(metadata["points"], dtype=np.float64) - metadata["x_reference_index"])
            * metadata["x_increment_s"]
        ) + metadata["x_zero_s"]

    return metadata["x_zero_s"] + np.arange(metadata["points"], dtype=np.float64) * metadata["x_increment_s"]


def build_voltage_axis(raw_codes, metadata):
    if "y_reference_code" in metadata:
        return (
            (raw_codes.astype(np.float32) - metadata["y_origin_code"] - metadata["y_reference_code"])
            * metadata["y_increment_v_per_code"]
        )

    return (
        (raw_codes.astype(np.float32) - metadata["y_offset_code"]) * metadata["y_multiplier_v_per_code"]
    ) + metadata["y_zero_v"]


def create_compressed_dataset(group, name, data):
    group.create_dataset(
        name,
        data=data,
        compression=HDF5_COMPRESSION,
        compression_opts=HDF5_COMPRESSION_OPTS,
        shuffle=True,
    )


def get_storage_metadata(raw_codes, metadata):
    stored = {
        "channel": metadata["channel"],
        "points": len(raw_codes),
        "raw_dtype": str(raw_codes.dtype),
        "x_increment_s": metadata["x_increment_s"],
        "x_zero_s": metadata["x_zero_s"],
    }

    if "x_reference_index" in metadata:
        stored["x_reference_index"] = metadata["x_reference_index"]

    if "y_reference_code" in metadata:
        stored["y_increment_v_per_code"] = metadata["y_increment_v_per_code"]
        stored["y_origin_code"] = metadata["y_origin_code"]
        stored["y_reference_code"] = metadata["y_reference_code"]
    else:
        stored["y_multiplier_v_per_code"] = metadata["y_multiplier_v_per_code"]
        stored["y_offset_code"] = metadata["y_offset_code"]
        stored["y_zero_v"] = metadata["y_zero_v"]

    return stored


def main():
    rm = pyvisa.ResourceManager()
    resource_name = find_single_usb_scope(rm)
    hdf5_path = build_hdf5_path()
    run_start = time.perf_counter()

    try:
        scope = rm.open_resource(resource_name)
        scope.timeout = 30000
        scope.chunk_size = 8 * 1024 * 1024

        try:
            instrument_idn = get_instrument_idn(scope)
            displayed_channels = get_displayed_channels(scope)
            results = []

            for channel in displayed_channels:
                raw_codes, metadata = read_waveform(scope, channel)
                results.append((channel, raw_codes, metadata))
        finally:
            scope.close()

        save_start = time.perf_counter()
        with h5py.File(hdf5_path, "w") as h5_file:
            h5_file.attrs["format"] = "pytek-waveform"
            h5_file.attrs["format_version"] = HDF5_FORMAT_VERSION
            h5_file.attrs["resource_name"] = resource_name
            h5_file.attrs["instrument_idn"] = instrument_idn
            h5_file.attrs["captured_at"] = datetime.now().isoformat(timespec="seconds")
            h5_file.attrs["displayed_channels"] = np.array(
                [f"CH{channel}" for channel in displayed_channels],
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

            channels_group = h5_file.create_group("channels")

            for channel, raw_codes, metadata in results:
                channel_group = channels_group.create_group(f"CH{channel}")

                create_compressed_dataset(channel_group, "raw_codes", raw_codes)

                for key, value in get_storage_metadata(raw_codes, metadata).items():
                    channel_group.attrs[key] = value

        save_seconds = time.perf_counter() - save_start

        print(f"Connected: {resource_name}")
        print(f"Instrument: {instrument_idn}")
        print(f"Displayed channels: {', '.join(f'CH{channel}' for channel in displayed_channels)}")

        for channel, _, metadata in results:
            print(
                f"CH{channel}: points={metadata['captured_points']} "
                f"(reported={metadata['reported_points']}), "
                f"transfer={metadata.get('transfer_seconds', 0.0):.3f} s, "
                f"avg={metadata['voltage_avg_v']:.6f} V, "
                f"max={metadata['voltage_max_v']:.6f} V, "
                f"min={metadata['voltage_min_v']:.6f} V"
            )

        print(f"Save time: {save_seconds:.3f} s")
        print(f"Total time: {time.perf_counter() - run_start:.3f} s")
        print(f"Saved: {hdf5_path.name}")
        print("Done")
    finally:
        rm.close()


if __name__ == "__main__":
    main()
