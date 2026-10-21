import sys
from .main import main
from sys import stderr


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("byeeeee...", file=stderr)
    except Exception as error:
        print(f"Unexpected error: {error}", file=stderr)
        sys.exit(1)
