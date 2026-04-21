import argparse
import math
from pathlib import Path

import h5py
import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            "A Qt binding is required. Install PySide6 or PyQt5 before running this viewer."
        ) from exc

try:
    import pyqtgraph as pg
except ImportError as exc:
    raise RuntimeError("pyqtgraph is required. Install it before running this viewer.") from exc


CHANNEL_COLORS = [
    (255, 210, 60),
    (85, 170, 255),
    (255, 120, 120),
    (120, 220, 160),
]
BUTTON_STYLE_TEMPLATE = """
QPushButton {{
    color: rgb{color};
    border: 1px solid rgb{color};
    border-radius: 4px;
    padding: 6px 10px;
    background: transparent;
    font-weight: 600;
}}
QPushButton:checked {{
    background: rgba{color_alpha};
}}
QPushButton:disabled {{
    color: rgb(120, 120, 120);
    border-color: rgb(120, 120, 120);
}}
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Preview a pytek waveform HDF5 file in PyQtGraph.")
    parser.add_argument("path", nargs="?", help="Path to a waveform HDF5 file.")
    return parser.parse_args()


def compute_time_axis(indices, metadata):
    if "x_reference_index" in metadata:
        return ((indices - metadata["x_reference_index"]) * metadata["x_increment_s"]) + metadata["x_zero_s"]
    return metadata["x_zero_s"] + (indices * metadata["x_increment_s"])


def compute_voltage_axis(raw_codes, metadata):
    if "y_reference_code" in metadata:
        return (
            (raw_codes.astype(np.float32) - metadata["y_origin_code"] - metadata["y_reference_code"])
            * metadata["y_increment_v_per_code"]
        )
    return (
        (raw_codes.astype(np.float32) - metadata["y_offset_code"]) * metadata["y_multiplier_v_per_code"]
    ) + metadata["y_zero_v"]


def compute_voltage_at_index(raw_code, metadata):
    return float(compute_voltage_axis(np.array([raw_code]), metadata)[0])


def build_downsampled_trace(raw_codes, start_index, metadata, target_buckets):
    bucket_size = max(1, math.ceil(len(raw_codes) / target_buckets))
    bucket_count = math.ceil(len(raw_codes) / bucket_size)
    selected_indices = []
    selected_codes = []

    for bucket_index in range(bucket_count):
        local_start = bucket_index * bucket_size
        local_stop = min(local_start + bucket_size, len(raw_codes))
        bucket = raw_codes[local_start:local_stop]
        if len(bucket) == 0:
            continue

        first_pos = 0
        last_pos = len(bucket) - 1
        min_pos = int(np.argmin(bucket))
        max_pos = int(np.argmax(bucket))

        positions = sorted({first_pos, min_pos, max_pos, last_pos})
        for position in positions:
            selected_indices.append(start_index + local_start + position)
            selected_codes.append(bucket[position])

    index_array = np.array(selected_indices, dtype=np.int64)
    code_array = np.array(selected_codes, dtype=raw_codes.dtype)
    return compute_time_axis(index_array, metadata), compute_voltage_axis(code_array, metadata), bucket_size


def decode_attrs(h5_attrs):
    return {key: h5_attrs[key].item() if hasattr(h5_attrs[key], "item") else h5_attrs[key] for key in h5_attrs}


def dataset_min_max(dataset):
    chunk_len = dataset.chunks[0] if dataset.chunks else min(len(dataset), 1_000_000)
    raw_min = None
    raw_max = None
    for start in range(0, len(dataset), chunk_len):
        chunk = dataset[start:start + chunk_len]
        if len(chunk) == 0:
            continue
        chunk_min = int(np.min(chunk))
        chunk_max = int(np.max(chunk))
        raw_min = chunk_min if raw_min is None else min(raw_min, chunk_min)
        raw_max = chunk_max if raw_max is None else max(raw_max, chunk_max)
    return raw_min, raw_max


def format_time_value(seconds, span_seconds):
    abs_span = abs(span_seconds)
    if abs_span < 1e-6:
        scale = 1e9
        unit = "ns"
    elif abs_span < 1e-3:
        scale = 1e6
        unit = "us"
    elif abs_span < 1:
        scale = 1e3
        unit = "ms"
    else:
        scale = 1.0
        unit = "s"
    return f"{seconds * scale:.2f} {unit}"


def format_voltage_value(voltage, span_volts):
    abs_span = abs(span_volts)
    if abs_span < 1e-3:
        scale = 1e6
        unit = "uV"
    elif abs_span < 1:
        scale = 1e3
        unit = "mV"
    else:
        scale = 1.0
        unit = "V"
    return f"{voltage * scale:.2f} {unit}"


def time_scale_and_unit(span_seconds):
    abs_span = abs(span_seconds)
    if abs_span < 1e-6:
        return 1e9, "ns"
    if abs_span < 1e-3:
        return 1e6, "us"
    if abs_span < 1:
        return 1e3, "ms"
    return 1.0, "s"


def format_time_with_scale(seconds, scale_and_unit):
    scale, unit = scale_and_unit
    return f"{seconds * scale:.2f} {unit}"


class RelativeTimeAxis(pg.AxisItem):
    def __init__(self, time_unit_getter, orientation="bottom"):
        super().__init__(orientation=orientation)
        self._time_unit_getter = time_unit_getter

    def tickStrings(self, values, scale, spacing):
        unit_scale, _unit = self._time_unit_getter()
        return [f"{value * unit_scale:.2f}" for value in values]

    def labelString(self):
        return "Time"

    def tickValues(self, minVal, maxVal, size):
        return super().tickValues(minVal, maxVal, size)


class ScopePlotWidget(pg.PlotWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_pen = QtGui.QPen(QtGui.QColor(90, 90, 90, 110))
        self.grid_pen.setWidth(1)
        self.center_pen = QtGui.QPen(QtGui.QColor(220, 220, 220, 210))
        self.center_pen.setWidth(1)

    def _major_tick_values(self, axis, lower, upper, size):
        tick_levels = axis.tickValues(lower, upper, size)
        for _spacing, values in tick_levels:
            if values:
                return values
        return []

    def _map_view_to_viewport_x(self, x_value):
        scene_point = self.getViewBox().mapViewToScene(pg.Point(float(x_value), 0.0))
        return self.mapFromScene(scene_point).x()

    def _map_view_to_viewport_y(self, y_value):
        scene_point = self.getViewBox().mapViewToScene(pg.Point(0.0, float(y_value)))
        return self.mapFromScene(scene_point).y()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QtGui.QPainter(self.viewport())
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        width = self.viewport().width()
        height = self.viewport().height()
        x_min, x_max = self.getViewBox().viewRange()[0]
        y_min, y_max = self.getViewBox().viewRange()[1]

        bottom_axis = self.getPlotItem().getAxis("bottom")
        left_axis = self.getPlotItem().getAxis("left")
        x_ticks = self._major_tick_values(bottom_axis, x_min, x_max, width)
        y_ticks = self._major_tick_values(left_axis, y_min, y_max, height)

        painter.setPen(self.grid_pen)
        for x_value in x_ticks:
            x = round(self._map_view_to_viewport_x(x_value))
            painter.drawLine(x, 0, x, height)

        for y_value in y_ticks:
            y = round(self._map_view_to_viewport_y(y_value))
            painter.drawLine(0, y, width, y)

        center_x = round(self._map_view_to_viewport_x(0.0))
        center_y = round(self._map_view_to_viewport_y((y_min + y_max) / 2.0))
        painter.setPen(self.center_pen)
        painter.drawLine(center_x, 0, center_x, height)
        painter.drawLine(0, center_y, width, center_y)
        painter.end()


class WaveformViewBox(pg.ViewBox):
    restoreRequested = QtCore.Signal()
    panRequested = QtCore.Signal(float)

    def wheelEvent(self, event, axis=None):
        delta = event.delta() if hasattr(event, "delta") else event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return

        x_min, x_max = self.viewRange()[0]
        width = max(1e-18, x_max - x_min)
        scale = 0.85 if delta > 0 else 1.0 / 0.85
        new_width = width * scale
        half_width = new_width / 2.0
        self.setXRange(-half_width, half_width, padding=0)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.restoreRequested.emit()
        event.accept()

    def mouseDragEvent(self, event, axis=None):
        if event.button() == QtCore.Qt.LeftButton:
            current_view = self.mapSceneToView(event.scenePos())
            last_view = self.mapSceneToView(event.lastScenePos())
            self.panRequested.emit(float(current_view.x() - last_view.x()))
            event.accept()
            return
        super().mouseDragEvent(event, axis=axis)


class WaveformViewer(QtWidgets.QMainWindow):
    def __init__(self, hdf5_path):
        super().__init__()
        self.hdf5_path = Path(hdf5_path)
        self.h5_file = h5py.File(self.hdf5_path, "r")
        self.channels_group = self.h5_file["channels"]
        self.channel_names = sorted(self.channels_group.keys())
        if not self.channel_names:
            raise RuntimeError("No channels were found in the HDF5 file.")

        self.channel_data = {}
        self.channel_buttons = {}
        self.focused_channel = None
        self.global_x_range = (0.0, 1.0)
        self.global_half_width = 1.0
        self.global_y_range = (-1.0, 1.0)
        self.crosshair_time = None
        self.display_offset_time = 0.0
        self.absolute_time_scale = (1.0, "s")

        self.setWindowTitle(f"pytek waveform viewer - {self.hdf5_path.name}")
        self.resize(1280, 760)

        central = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(QtWidgets.QLabel("Channels"))
        self.button_bar = QtWidgets.QHBoxLayout()
        self.button_bar.setSpacing(8)
        top_bar.addLayout(self.button_bar)
        top_bar.addStretch(1)

        content_layout = QtWidgets.QHBoxLayout()

        self.time_axis = RelativeTimeAxis(self.current_time_unit, orientation="bottom")
        self.view_box = WaveformViewBox()
        self.view_box.setMouseMode(pg.ViewBox.PanMode)
        self.view_box.setMouseEnabled(x=True, y=False)
        self.view_box.sigXRangeChanged.connect(self.on_view_changed)
        self.view_box.restoreRequested.connect(self.restore_global_view)
        self.view_box.panRequested.connect(self.on_pan_requested)

        self.plot_widget = ScopePlotWidget(viewBox=self.view_box, axisItems={"bottom": self.time_axis})
        self.plot_widget.setLabel("left", "Voltage", units="V")
        self.plot_widget.setLabel("bottom", "Relative Time")
        self.plot_widget.setClipToView(True)
        self.plot_widget.setDownsampling(auto=False)
        self.plot_widget.getPlotItem().setMenuEnabled(False)
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.legend = self.plot_widget.addLegend(offset=(10, 10))

        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((200, 200, 200), width=1))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((200, 200, 200), width=1))
        self.plot_widget.addItem(self.v_line, ignoreBounds=True)
        self.plot_widget.addItem(self.h_line, ignoreBounds=True)

        self.cursor_panel = QtWidgets.QFrame()
        self.cursor_panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.cursor_panel.setMinimumWidth(240)
        panel_layout = QtWidgets.QVBoxLayout(self.cursor_panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.addWidget(QtWidgets.QLabel("Cursor"))
        self.time_label = QtWidgets.QLabel("dt: --")
        panel_layout.addWidget(self.time_label)
        self.absolute_time_label = QtWidgets.QLabel("t: --")
        panel_layout.addWidget(self.absolute_time_label)
        self.value_labels = {}
        for channel_name in self.channel_names:
            label = QtWidgets.QLabel(f"{channel_name}: --")
            self.value_labels[channel_name] = label
            panel_layout.addWidget(label)
        panel_layout.addStretch(1)

        content_layout.addWidget(self.plot_widget, 1)
        content_layout.addWidget(self.cursor_panel)

        root_layout.addLayout(top_bar)
        root_layout.addLayout(content_layout, 1)
        self.setCentralWidget(central)

        self.load_channels()
        self.restore_global_view()

    def closeEvent(self, event):
        self.h5_file.close()
        super().closeEvent(event)

    def current_time_unit(self):
        x_min, x_max = self.view_box.viewRange()[0]
        return time_scale_and_unit(x_max - x_min)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_plot()

    def load_channels(self):
        overall_start = None
        overall_stop = None
        overall_y_min = None
        overall_y_max = None

        for index, channel_name in enumerate(self.channel_names):
            channel_group = self.channels_group[channel_name]
            metadata = decode_attrs(channel_group.attrs)
            dataset = channel_group["raw_codes"]
            points = int(metadata["points"])
            color = CHANNEL_COLORS[index % len(CHANNEL_COLORS)]

            full_time = compute_time_axis(np.array([0, max(0, points - 1)], dtype=np.float64), metadata)
            overall_start = full_time[0] if overall_start is None else min(overall_start, float(full_time[0]))
            overall_stop = full_time[1] if overall_stop is None else max(overall_stop, float(full_time[1]))

            raw_min, raw_max = dataset_min_max(dataset)
            voltage_min = compute_voltage_at_index(raw_min, metadata)
            voltage_max = compute_voltage_at_index(raw_max, metadata)
            overall_y_min = voltage_min if overall_y_min is None else min(overall_y_min, voltage_min)
            overall_y_max = voltage_max if overall_y_max is None else max(overall_y_max, voltage_max)

            pen = pg.mkPen(color=color, width=1)
            curve = self.plot_widget.plot(name=channel_name, pen=pen)
            button = QtWidgets.QPushButton(channel_name)
            button.setCheckable(True)
            button.setStyleSheet(
                BUTTON_STYLE_TEMPLATE.format(color=color, color_alpha=(color[0], color[1], color[2], 45))
            )
            button.clicked.connect(lambda checked, name=channel_name: self.on_channel_button_clicked(name))
            self.button_bar.addWidget(button)

            self.channel_buttons[channel_name] = button
            self.channel_data[channel_name] = {
                "dataset": dataset,
                "metadata": metadata,
                "points": points,
                "curve": curve,
                "visible": True,
                "color": color,
            }

        half_span = max(abs(float(overall_start)), abs(float(overall_stop)))
        self.global_half_width = half_span
        self.global_x_range = (-half_span, half_span)
        self.absolute_time_scale = time_scale_and_unit(abs(float(overall_stop - overall_start)))
        if overall_y_min == overall_y_max:
            margin = max(0.1, abs(overall_y_min) * 0.05 + 1e-6)
        else:
            margin = (overall_y_max - overall_y_min) * 0.08
        self.global_y_range = (overall_y_min - margin, overall_y_max + margin)
        self.focused_channel = self.channel_names[0]

        for channel_name in self.channel_names:
            self.set_channel_visible(channel_name, True)
            self.channel_buttons[channel_name].setChecked(channel_name == self.focused_channel)

        self.raise_focused_channel()
        self.plot_widget.setTitle(f"{self.hdf5_path.name} - {max(info['points'] for info in self.channel_data.values()):,} max points")

    def on_channel_button_clicked(self, channel_name):
        if self.focused_channel == channel_name and self.channel_data[channel_name]["visible"]:
            self.set_channel_visible(channel_name, False)
            self.channel_buttons[channel_name].setChecked(False)
            if self.focused_channel == channel_name:
                self.focused_channel = next(
                    (name for name in self.channel_names if self.channel_data[name]["visible"]),
                    None,
                )
        else:
            self.set_channel_visible(channel_name, True)
            self.focused_channel = channel_name

        for name, button in self.channel_buttons.items():
            button.blockSignals(True)
            button.setChecked(name == self.focused_channel and self.channel_data[name]["visible"])
            button.blockSignals(False)

        self.raise_focused_channel()
        self.refresh_plot()

    def set_channel_visible(self, channel_name, visible):
        self.channel_data[channel_name]["visible"] = visible
        self.channel_data[channel_name]["curve"].setVisible(visible)
        self.value_labels[channel_name].setVisible(visible)

    def raise_focused_channel(self):
        base_z = 10
        for order, channel_name in enumerate(self.channel_names):
            z_value = base_z + order
            if channel_name == self.focused_channel and self.channel_data[channel_name]["visible"]:
                z_value = base_z + len(self.channel_names) + 1
            self.channel_data[channel_name]["curve"].setZValue(z_value)

    def restore_global_view(self):
        self.display_offset_time = 0.0
        self.view_box.setXRange(*self.global_x_range, padding=0)
        self.view_box.setYRange(*self.global_y_range, padding=0)
        self.refresh_plot()

    def on_view_changed(self, *_args):
        self.refresh_plot()

    def on_pan_requested(self, delta_x):
        self.display_offset_time -= delta_x
        self.refresh_plot()

    def get_reference_channel(self):
        if self.focused_channel and self.channel_data[self.focused_channel]["visible"]:
            return self.focused_channel

        for channel_name in self.channel_names:
            if self.channel_data[channel_name]["visible"]:
                return channel_name

        return None

    def nearest_sample_time(self, relative_time):
        reference_channel = self.get_reference_channel()
        if reference_channel is None:
            return None, None

        channel_info = self.channel_data[reference_channel]
        metadata = channel_info["metadata"]
        point_count = channel_info["points"]
        actual_time = relative_time + self.display_offset_time

        if "x_reference_index" in metadata:
            sample_index = int(round(((actual_time - metadata["x_zero_s"]) / metadata["x_increment_s"]) + metadata["x_reference_index"]))
        else:
            sample_index = int(round((actual_time - metadata["x_zero_s"]) / metadata["x_increment_s"]))

        sample_index = max(0, min(point_count - 1, sample_index))
        absolute_sample_time = float(compute_time_axis(np.array([sample_index], dtype=np.float64), metadata)[0])
        relative_sample_time = absolute_sample_time - self.display_offset_time
        return sample_index, relative_sample_time

    def visible_index_range(self, metadata, total_points):
        x_min, x_max = self.view_box.viewRange()[0]
        x_min += self.display_offset_time
        x_max += self.display_offset_time
        x_increment = float(metadata["x_increment_s"])

        if "x_reference_index" in metadata:
            x_reference = float(metadata["x_reference_index"])
            start = int(math.floor(((x_min - metadata["x_zero_s"]) / x_increment) + x_reference))
            stop = int(math.ceil(((x_max - metadata["x_zero_s"]) / x_increment) + x_reference))
        else:
            start = int(math.floor((x_min - metadata["x_zero_s"]) / x_increment))
            stop = int(math.ceil((x_max - metadata["x_zero_s"]) / x_increment))

        start = max(0, start)
        stop = min(total_points, max(start + 1, stop))
        return start, stop

    def refresh_plot(self):
        plot_width = max(200, self.plot_widget.width())
        target_buckets = max(100, plot_width // 2)

        for channel_name, channel_info in self.channel_data.items():
            if not channel_info["visible"]:
                channel_info["curve"].setData([], [])
                continue

            metadata = channel_info["metadata"]
            start, stop = self.visible_index_range(metadata, channel_info["points"])
            visible_points = max(1, stop - start)
            raw_codes = channel_info["dataset"][start:stop]

            if visible_points > target_buckets * 2:
                x_values, y_values, _bucket_size = build_downsampled_trace(raw_codes, start, metadata, target_buckets)
            else:
                indices = np.arange(start, stop, dtype=np.float64)
                x_values = compute_time_axis(indices, metadata)
                y_values = compute_voltage_axis(raw_codes, metadata)

            channel_info["curve"].setData(x=x_values - self.display_offset_time, y=y_values)
        self.update_cursor_readout()

    def on_mouse_moved(self, scene_pos):
        if not self.plot_widget.sceneBoundingRect().contains(scene_pos):
            return

        mouse_point = self.view_box.mapSceneToView(scene_pos)
        _sample_index, snapped_time = self.nearest_sample_time(float(mouse_point.x()))
        if snapped_time is None:
            return

        self.crosshair_time = snapped_time
        self.v_line.setPos(self.crosshair_time)
        self.h_line.setPos(float(mouse_point.y()))
        self.update_cursor_readout()

    def update_cursor_readout(self):
        if self.crosshair_time is None:
            self.time_label.setText("dt: --")
            self.absolute_time_label.setText("t: --")
            for channel_name, label in self.value_labels.items():
                label.setText(f"{channel_name}: --")
            return

        relative_time = self.crosshair_time
        time_scale, time_unit = self.current_time_unit()
        self.time_label.setText(f"dt: {relative_time * time_scale:.2f} {time_unit}")
        actual_time = self.crosshair_time + self.display_offset_time
        self.absolute_time_label.setText(f"t: {format_time_with_scale(actual_time, self.absolute_time_scale)}")
        y_min, y_max = self.view_box.viewRange()[1]
        voltage_span = y_max - y_min

        for channel_name, channel_info in self.channel_data.items():
            label = self.value_labels[channel_name]
            metadata = channel_info["metadata"]
            point_count = channel_info["points"]

            if "x_reference_index" in metadata:
                index = int(round(((actual_time - metadata["x_zero_s"]) / metadata["x_increment_s"]) + metadata["x_reference_index"]))
            else:
                index = int(round((actual_time - metadata["x_zero_s"]) / metadata["x_increment_s"]))

            if 0 <= index < point_count:
                raw_code = channel_info["dataset"][index]
                voltage = compute_voltage_at_index(raw_code, metadata)
                label.setText(f"{channel_name}: {format_voltage_value(voltage, voltage_span)}")
            else:
                label.setText(f"{channel_name}: --")


def main():
    args = parse_args()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    if args.path:
        hdf5_path = Path(args.path)
    else:
        hdf5_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Open pytek waveform file",
            "",
            "HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if not hdf5_path:
            return

    pg.setConfigOptions(antialias=False)
    viewer = WaveformViewer(hdf5_path)
    viewer.show()
    app.exec()


if __name__ == "__main__":
    main()
