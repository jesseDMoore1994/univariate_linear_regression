import argparse
import sys
import logging

_logger = logging.getLogger(__name__)


def setup_logging():
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main():
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    setup_logging()


if __name__ == "__main__":
    main(sys.argv[1:])
