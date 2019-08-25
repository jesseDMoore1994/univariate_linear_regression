import argparse
import sys
import logging
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_logger = logging.getLogger(__name__)


def random_array_of_size(size, minimum=1, maximum=25):
    return [ random.randint(minimum, maximum) for i in range(size) ]


def create_plot(data):
    df = pd.DataFrame(data, columns= ['x', 'y'])
    df.plot.scatter(x='x', y='y')
    plt.savefig('/app/plot.png')


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
    x = [ i for i in range(1, 11) ]
    y = random_array_of_size(10)
    data = { 'x': x, 'y': y}
    create_plot(data)


if __name__ == "__main__":
    main(sys.argv[1:])
