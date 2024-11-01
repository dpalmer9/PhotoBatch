import sys
from PySide6.QtWidgets import QApplication
from GUI.interface import FiberPhotometryApp

def launch_photobatch():
    app = QApplication(sys.argv)
    window = FiberPhotometryApp()
    window.show()
    sys.exit(app.exec())