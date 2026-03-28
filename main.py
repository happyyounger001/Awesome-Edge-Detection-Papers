from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import FencingMainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = FencingMainWindow(config_path="config.yaml")
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
