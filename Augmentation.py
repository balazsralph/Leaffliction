import sys
from pathlib import Path

def main() -> None:
    if len(sys.argv) != 2:
        print("Wrong number of arguments")
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.is_dir():
        print("Not a directory")
        sys.exit(1)
        
   
if __name__ == "__main__":
    main()