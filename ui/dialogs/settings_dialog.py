from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
)


class SettingsDialog(QDialog):
    def __init__(self, config, on_apply, on_reset, parent=None):
        super().__init__(parent)
        self._config = config
        self._on_apply = on_apply
        self._on_reset = on_reset
        self.setWindowTitle("参数设置")

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.stability_spin = QDoubleSpinBox()
        self.stability_spin.setRange(0.1, 5.0)
        self.stability_spin.setDecimals(2)
        self.stability_spin.setSuffix(" 秒")
        form.addRow("弓步稳定性：", self.stability_spin)

        self.thigh_angle_spin = QDoubleSpinBox()
        self.thigh_angle_spin.setRange(1.0, 90.0)
        self.thigh_angle_spin.setDecimals(1)
        self.thigh_angle_spin.setSuffix(" 度")
        form.addRow("抬大腿角度阈值：", self.thigh_angle_spin)

        self.head_tilt_spin = QDoubleSpinBox()
        self.head_tilt_spin.setRange(1.0, 45.0)
        self.head_tilt_spin.setDecimals(1)
        self.head_tilt_spin.setSuffix(" 度")
        form.addRow("头部偏斜阈值：", self.head_tilt_spin)

        self.knee_below_hip_checkbox = QCheckBox("膝关节点需低于髋关节点")
        form.addRow("膝髋位置规则：", self.knee_below_hip_checkbox)
        layout.addLayout(form)

        row = QHBoxLayout()
        apply_btn = QPushButton("应用参数")
        reset_btn = QPushButton("恢复默认")
        apply_btn.clicked.connect(self._apply)
        reset_btn.clicked.connect(self._reset)
        row.addWidget(apply_btn)
        row.addWidget(reset_btn)
        layout.addLayout(row)

        self.reload_from_config()

    def reload_from_config(self):
        self.stability_spin.setValue(self._config.stability_seconds_threshold)
        self.thigh_angle_spin.setValue(self._config.thigh_raise_angle_threshold)
        self.head_tilt_spin.setValue(self._config.head_tilt_angle_threshold)
        self.knee_below_hip_checkbox.setChecked(self._config.require_knee_below_hip_for_thigh_raise)

    def _apply(self):
        self._on_apply(
            self.stability_spin.value(),
            self.thigh_angle_spin.value(),
            self.head_tilt_spin.value(),
            self.knee_below_hip_checkbox.isChecked(),
        )

    def _reset(self):
        self._on_reset()
        self.reload_from_config()
