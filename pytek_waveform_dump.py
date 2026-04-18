from array import array
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pyvisa


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


def configure_waveform_transfer(scope, channel):
    scope.write("HEADER 0")
    scope.write(f"DATA:SOURCE CH{channel}")
    scope.write("DATA:START 1")
    scope.write("DATA:STOP 5000000")
    scope.write("DATA:ENCDG RIBINARY")
    scope.write("DATA:WIDTH 1")


def read_waveform(scope, channel):
    configure_waveform_transfer(scope, channel)

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


rm = pyvisa.ResourceManager()
resource_name = find_single_usb_scope(rm)
hdf5_path = build_hdf5_path()

scope = rm.open_resource(resource_name)
scope.timeout = 30000
scope.chunk_size = 1024000

try:
    displayed_channels = get_displayed_channels(scope)
    results = []

    for channel in displayed_channels:
        raw_codes, metadata = read_waveform(scope, channel)
        results.append((channel, raw_codes, metadata))
finally:
    scope.close()

with h5py.File(hdf5_path, "w") as h5_file:
    h5_file.attrs["resource_name"] = resource_name
    h5_file.attrs["captured_at"] = datetime.now().isoformat(timespec="seconds")
    h5_file.attrs["displayed_channels"] = np.array(
        [f"CH{channel}" for channel in displayed_channels],
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    channels_group = h5_file.create_group("channels")

    for channel, raw_codes, metadata in results:
        channel_group = channels_group.create_group(f"CH{channel}")

        channel_group.create_dataset(
            "raw_codes",
            data=raw_codes,
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )

        time_axis = metadata["x_zero_s"] + np.arange(metadata["points"], dtype=np.float64) * metadata["x_increment_s"]
        voltage_axis = ((raw_codes.astype(np.float32) - metadata["y_offset_code"]) * metadata["y_multiplier_v_per_code"]) + metadata["y_zero_v"]

        channel_group.create_dataset(
            "time_s",
            data=time_axis,
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )
        channel_group.create_dataset(
            "voltage_v",
            data=voltage_axis,
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )

        for key, value in metadata.items():
            channel_group.attrs[key] = value

print(f"Connected: {resource_name}")
print(f"Displayed channels: {', '.join(f'CH{channel}' for channel in displayed_channels)}")

for channel, _, metadata in results:
    print(
        f"CH{channel}: avg={metadata['voltage_avg_v']:.6f} V, "
        f"max={metadata['voltage_max_v']:.6f} V, "
        f"min={metadata['voltage_min_v']:.6f} V"
    )

print(f"Saved: {hdf5_path.name}")
print("Done")
