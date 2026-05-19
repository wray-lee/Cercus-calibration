"""
Cercus-Calibrator — entry point.

Run:  python main.py
"""

from src.core.ui import CercusCalibratorUI


def main():
    app = CercusCalibratorUI()
    app.mainloop()


if __name__ == "__main__":
    main()
