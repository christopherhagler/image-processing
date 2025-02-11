import os
import sys
import logging
from homework import run_homework_steps


def setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')


def main():
    debug = os.getenv("DEBUG", False)
    setup_logging(debug=debug)
    run_homework_steps()
    return 0


if __name__ == '__main__':
    sys.exit(main())
