import argparse
import sys
import logging
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_logger = logging.getLogger(__name__)


def predictor(x, bias, weight):
    return bias + weight*x


def cost_function(X, Y, bias, weight):
    return (1 / 2 * len(X)) * sum([(predictor(x, bias, weight) - y)**2 for x, y in zip(X, Y)])


def cost_derived_with_respect_to_bias(x, y, bias, weight):
    bias_derive = (predictor(x, bias, weight) - y)
    return bias_derive


def cost_derived_with_respect_to_weight(x, y, bias, weight):
    weight_derive = x * (predictor(x, bias, weight) - y)
    return weight_derive


# use the gradient of the cost function to get new values with respect to the learning rate
def update_weight_and_bias(x, y, bias, weight, learning_rate):
    d_db = sum([cost_derived_with_respect_to_bias(a, b, bias, weight) for a, b in zip(x, y)])
    d_dw = sum([cost_derived_with_respect_to_weight(a, b, bias, weight) for a, b in zip(x, y)])

    # We subtract because the derivatives point in direction of steepest ascent
    b = bias - (learning_rate * (d_db / len(x)) * learning_rate)
    w = weight - (learning_rate * (d_dw / len(x)) * learning_rate)

    return w, b

def univariate_linear_regression(X, Y):
    # guess initial weight and bias
    weight = Y[0]
    bias = Y[1] - Y[0] / X[1] - X[0]
    learning_rate = .01
    iters = 50

    for i in range(iters):
        weight,bias = update_weight_and_bias(X, Y, weight, bias, learning_rate)

        #Calculate cost for auditing purposes
        cost = cost_function(X, Y, weight, bias)

        # Log Progress
        if i % 10 == 0:
            print("iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2}".format(i, weight, bias, cost))


    return weight, bias



def unconstrained_optimization(x, y):
    assert len(x) == len(y)
    m = len(x)
    xx = [a**2 for a in x]
    xy = [a*b for a, b in zip(x,y)]
    sum_x = sum(x)
    sum_xx = sum(xx)
    sum_y = sum(y)
    sum_xy = sum(xy)
    v0 = ((sum_xx * sum_y) - (sum_x * sum_xy)) / ((m * sum_xx) - ((sum_x)**2))
    v1 = ((m * sum_xy) - (sum_x * sum_y)) / ((m * sum_xx) - (sum_x)**2)
    print(f"bias: {v0}; weight: {v1}")
    return v0, v1


def create_best_fit_bias_and_weight(data):
    return unconstrained_optimization(data['x'], data['y'])


def create_best_guess_bias_and_weight(data):
    return univariate_linear_regression(data['x'], data['y'])


def random_array_of_size(size, minimum=1, maximum=100):
    return [ random.randint(minimum, maximum) for i in range(size) ]


def make_correlated_sequence(x, y, rho=.95):
    return [ (rho * a) + (math.sqrt(1 - rho**2) * b) for a,b in zip(x,y) ]


def create_line(x, v0, v1):
    return [(x, v0+v1*x) for x in x]


def create_plot(scatter_data):
    # create best fit algorithmically
    b, w = create_best_fit_bias_and_weight(scatter_data)
    x, y = zip(*create_line(scatter_data['x'], b, w))
    best_fit_line_data = {'x': x, 'y': y}
    # create best fit using ML
    b, w = create_best_guess_bias_and_weight(scatter_data)
    x, y = zip(*create_line(scatter_data['x'], b, w))
    regression_line_data = {'x': x, 'y': y}
    # actually graph the stuff
    plt.plot(scatter_data['x'], scatter_data['y'], '.')
    plt.plot(best_fit_line_data['x'], best_fit_line_data['y'], 'g-')
    plt.plot(regression_line_data['x'], regression_line_data['y'], 'b-')
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
    x = [ i for i in range(1, 101) ]
    y = make_correlated_sequence(x, random_array_of_size(100))
    data = { 'x': x, 'y': y}
    create_plot(data)


if __name__ == "__main__":
    main(sys.argv[1:])
