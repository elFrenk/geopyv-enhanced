from pathlib import Path
import sys

def main():
    print("Python:", sys.executable)
    print("CWD:", Path.cwd())

    import geopyv
    print("geopyv imported from:", geopyv.__file__)

if __name__ == "__main__":
    main()