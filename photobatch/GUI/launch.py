import sys

def launch_photobatch():
    from PySide6.QtWidgets import QApplication

    from photobatch.GUI.interface import FiberPhotometryApp

    app = QApplication(sys.argv)
    window = FiberPhotometryApp()
    window.show()
    sys.exit(app.exec())