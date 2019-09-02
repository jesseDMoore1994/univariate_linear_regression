import argparse
import sys
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

logging.basicConfig(filename='linear_regression.log', filemode='w', format='%(levelname)s - %(message)s', level=logging.INFO)


# predictors
def simple_predictor(x, v0, v1):
    return v0 + v1*x


def multi_predictor(x1, x2, v0, v1, v2):
    return v0 + v1*x1 + v2*x2


#cost functions
def simple_cost_function(X, Y, v0, v1):
    return (1 / 2 * len(X)) * sum([(simple_predictor(x, v0, v1) - y)**2 for x, y in zip(X, Y)])


def multi_cost_function(X1, X2, Y, v0, v1, v2):
    return (1 / 2 * len(X1)) * sum([(multi_predictor(x1, x2, v0, v1, v2) - y)**2 for x1, x2, y in zip(X1, X2, Y)])


#partial derivatives for gradients
def simple_cost_derived_with_respect_to_v0(x, y, v0, v1):
    return simple_predictor(x, v0, v1) - y


def simple_cost_derived_with_respect_to_v1(x, y, v0, v1):
    return x * (simple_predictor(x, v0, v1) - y)


def multi_cost_derived_with_respect_to_v0(x1, x2, y, v0, v1, v2):
    return multi_predictor(x1, x2, v0, v1, v2) - y


def multi_cost_derived_with_respect_to_v1(x1, x2, y, v0, v1, v2):
    return x1 * (multi_predictor(x1, x2, v0, v1, v2) - y)


def multi_cost_derived_with_respect_to_v2(x1, x2, y, v0, v1, v2):
    return x2 * (multi_predictor(x1, x2, v0, v1, v2) - y)


# use the gradient of the cost function to get new values with respect to the learning rate
def update_vs_for_simple_linear_regression(x, y, v0, v1, learning_rate):
    d_dv0 = sum([simple_cost_derived_with_respect_to_v0(a, b, v0, v1) for a, b in zip(x, y)])
    d_dv1 = sum([simple_cost_derived_with_respect_to_v1(a, b, v0, v1) for a, b in zip(x, y)])

    # We subtract because the derivatives point in direction of steepest ascent
    v0 = v0 - (learning_rate * (d_dv0 / len(x)) * learning_rate)
    v1 = v1 - (learning_rate * (d_dv1 / len(x)) * learning_rate)

    return v1, v0


def update_vs_for_multiple_linear_regression(x1, x2, y, v0, v1, v2, learning_rate):
    d_dv0 = sum([multi_cost_derived_with_respect_to_v0(a, b, c, v0, v1, v2) for a, b, c in zip(x1, x2, y)])
    d_dv1 = sum([multi_cost_derived_with_respect_to_v1(a, b, c, v0, v1, v2) for a, b, c in zip(x1, x2, y)])
    d_dv2 = sum([multi_cost_derived_with_respect_to_v2(a, b, c, v0, v1, v2) for a, b, c in zip(x1, x2, y)])

    # We subtract because the derivatives point in direction of steepest ascent
    v0 = v0 - (learning_rate * (d_dv0 / len(x1)) * learning_rate)
    v1 = v1 - (learning_rate * (d_dv1 / len(x1)) * learning_rate)
    v2 = v2 - (learning_rate * (d_dv2 / len(x1)) * learning_rate)

    return v2, v1, v0


def simple_linear_regression(X, Y, alpha_init=.25, convergence=.01):
    logging.info(f"-----------------------Starting Simple Linear Regression----------------------------------")
    v1 = Y[0]
    v0 = Y[1] - Y[0] / X[1] - X[0]
    learning_rate = alpha_init

    logging.info(f"Initializing simple linear regression with v1={v1}, v0={v0}, learning rate={learning_rate}")
    logging.info(f"Algorithm is considered converged when difference between costs is less than {convergence}")
    costs = []
    i = 0
    cost = simple_cost_function(X, Y, v0, v1)
    logging.info(f"Initial cost={cost}")
    costs.append(cost)
    i = i + 1

    initial_values = {
        'X': X,
        'Y': Y,
        'v1': v1,
        'v0': v0,
        'learning_rate': learning_rate,
        'convergence': convergence
    }

    while True:
        v1, v0 = update_vs_for_simple_linear_regression(X, Y, v1, v0, learning_rate)

        #Calculate cost for auditing purposes
        cost = simple_cost_function(X, Y, v0, v1)
        costs.append(cost)
        logging.info(f"Update for iteration {i}. New values are v1={v1}, v0={v0}.")
        logging.info(f"Cost for iteration {i} is {cost}. This is {abs(costs[i] - costs[i-1])}")
        logging.info(f"difference from {i-1} iteration cost")

        if abs(costs[i] - costs[i-1]) < convergence:
            logging.info(f"Convergence reached, returning parameters and costs.")
            break

        i = i + 1

    output_values = {
        'v1': v1,
        'v0': v0,
        'costs': costs
    }

    return_vals = {
        'in': initial_values,
        'out': output_values
    }

    return return_vals


def multiple_linear_regression(X1, X2, Y, alpha_init=.001, convergence=.001):
    logging.info(f"-----------------------Starting Multiple Linear Regression--------------------------------")
    v0 = 0
    v1 = 1
    v2 = 1
    learning_rate = alpha_init

    logging.info(f"Initializing multiple linear regression with v2={v2}, v1={v1}, v0={v0}, learning rate={learning_rate}")
    logging.info(f"Algorithm is considered converged when difference between costs is less than {convergence}")
    costs = []
    i = 0
    cost = multi_cost_function(X1, X2, Y, v0, v1, v2)
    logging.info(f"Initial cost={cost}")
    costs.append(cost)
    i = i + 1

    initial_values = {
        'X1': X1,
        'X2': X2,
        'Y': Y,
        'v2': v2,
        'v1': v1,
        'v0': v0,
        'learning_rate': learning_rate,
        'convergence': convergence
    }

    while True:
        v2, v1, v0 = update_vs_for_multiple_linear_regression(X1, X2, Y, v0, v1, v2, learning_rate)

        #Calculate cost for auditing purposes
        cost = multi_cost_function(X1, X2, Y, v0, v1, v2)
        costs.append(cost)
        logging.info(f"Update for iteration {i}. New values are v2={v2}, v1={v1}, v0={v0}.")
        logging.info(f"Cost for iteration {i} is {cost}. This is {abs(costs[i] - costs[i-1])}")
        logging.info(f"difference from {i-1} iteration cost")

        if abs(costs[i] - costs[i-1]) < convergence:
            logging.info(f"Convergence reached, returning parameters and costs.")
            break

        i = i + 1

    output_values = {
        'v2': v2,
        'v1': v1,
        'v0': v0,
        'costs': costs
    }

    return_vals = {
        'in': initial_values,
        'out': output_values
    }


    return return_vals


def create_line(x, v0, v1):
    return [(x, v0+v1*x) for x in x]


def create_plane(x1, x2, v0, v1, v2):
    print(x1)
    print(x2)
    return [(a, b, v0+v1*a+v2*b) for a in x1 for b in x2]


def create_simple_linear_regression_plots():
    #solve the simple linear regression
    x = [ 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 ]
    y = [ 5.1, 6.1, 6.9, 7.8, 9.2, 9.9, 11.5, 12.0, 12.8 ]
    logging.info(f"Simple linear regression training data")
    logging.info(f"X: {x}")
    logging.info(f"Y: {y}")
    scatter_data = { 'x': x, 'y': y}

    # create best fit using ML
    simple_linear_reg_info = simple_linear_regression(scatter_data['x'], scatter_data['y'])
    v0, v1, costs = (
        simple_linear_reg_info['out']['v0'],
        simple_linear_reg_info['out']['v1'],
        simple_linear_reg_info['out']['costs']
    )
    logging.info(f"Simple linear regression returned the following parameters")
    logging.info(f"v0: {v0}")
    logging.info(f"v1: {v1}")
    logging.info(f"costs: {costs}")
    x, y = zip(*create_line(scatter_data['x'], v0, v1))
    regression_line_data = {'x': x, 'y': y}

    #plot cost as a measure of iterations
    x, y = zip(*list(enumerate(costs, 1)))
    cost_per_iter_line_data = {'x': x, 'y': y}

    # actually graph the stuff
    plt.figure()
    plt.plot(scatter_data['x'], scatter_data['y'], '.')
    plt.plot(regression_line_data['x'], regression_line_data['y'], 'b-', label='simple linear regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    logging.info(f"Writing linear regression graph to simple_linear_regression.png")
    plt.savefig('simple_linear_regression.png')
    plt.figure()
    plt.plot(cost_per_iter_line_data['x'], cost_per_iter_line_data['y'], 'b-', label='Cost for iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (J)')
    plt.legend(loc='upper right')
    logging.info(f"Writing cost over time for linear regression graph to simple_linear_regression_cost_per_iteration.png")
    plt.savefig('simple_linear_regression_cost_per_iteration.png')

    return simple_linear_reg_info


def create_multiple_linear_regression_plots():
    mlr_returns = []

    #solve the multiple linear regression
    x1 = [ 0, 1, 1, 2, 1 ]
    x2 = [ 1, 0, 1, 1, 2 ]
    y = [ 0.05, 2.05, 1.05, 1.95, -0.05 ]
    logging.info(f"Multiple linear regression training data:")
    logging.info(f"X1: {x1}")
    logging.info(f"X2: {x2}")
    logging.info(f"Y: {y}")
    scatter_data = { 'x1': x1, 'x2': x2, 'y': y}

    # create best fit using ML
    logging.info(f"Calling multiple linear regression with learning rate=0.001:")
    multi_linear_reg_info = multiple_linear_regression(scatter_data['x1'], scatter_data['x2'], scatter_data['y'])
    mlr_returns.append(multi_linear_reg_info)
    v0, v1, v2, costs1 = (
        multi_linear_reg_info['out']['v0'],
        multi_linear_reg_info['out']['v1'],
        multi_linear_reg_info['out']['v2'],
        multi_linear_reg_info['out']['costs']
    )
    x, y = zip(*list(enumerate(costs1, 1)))
    logging.info(f"Multiple linear regression returned the following parameters:")
    logging.info(f"v0: {v0}")
    logging.info(f"v1: {v1}")
    logging.info(f"v2: {v2}")
    costs1_per_iter_line_data = {'x': x, 'y': y}

    logging.info(f"Calling multiple linear regression with learning rate=0.01:")
    multi_linear_reg_info = multiple_linear_regression(scatter_data['x1'], scatter_data['x2'], scatter_data['y'], alpha_init=0.01)
    mlr_returns.append(multi_linear_reg_info)
    v0, v1, v2, costs2 = (
        multi_linear_reg_info['out']['v0'],
        multi_linear_reg_info['out']['v1'],
        multi_linear_reg_info['out']['v2'],
        multi_linear_reg_info['out']['costs']
    )
    x, y = zip(*list(enumerate(costs2, 1)))
    logging.info(f"Multiple linear regression returned the following parameters:")
    logging.info(f"v0: {v0}")
    logging.info(f"v1: {v1}")
    logging.info(f"v2: {v2}")
    costs2_per_iter_line_data = {'x': x, 'y': y}

    logging.info(f"Calling multiple linear regression with learning rate=0.1:")
    multi_linear_reg_info = multiple_linear_regression(scatter_data['x1'], scatter_data['x2'], scatter_data['y'], alpha_init=0.1)
    mlr_returns.append(multi_linear_reg_info)
    v0, v1, v2, costs3 = (
        multi_linear_reg_info['out']['v0'],
        multi_linear_reg_info['out']['v1'],
        multi_linear_reg_info['out']['v2'],
        multi_linear_reg_info['out']['costs']
    )
    x, y = zip(*list(enumerate(costs3, 1)))
    logging.info(f"Multiple linear regression returned the following parameters:")
    logging.info(f"v0: {v0}")
    logging.info(f"v1: {v1}")
    logging.info(f"v2: {v2}")
    costs3_per_iter_line_data = {'x': x, 'y': y}

    logging.info(f"Calling multiple linear regression with learning rate=0.2:")
    multi_linear_reg_info = multiple_linear_regression(scatter_data['x1'], scatter_data['x2'], scatter_data['y'], alpha_init=0.2)
    mlr_returns.append(multi_linear_reg_info)
    v0, v1, v2, costs4 = (
        multi_linear_reg_info['out']['v0'],
        multi_linear_reg_info['out']['v1'],
        multi_linear_reg_info['out']['v2'],
        multi_linear_reg_info['out']['costs']
    )
    x, y = zip(*list(enumerate(costs4, 1)))
    logging.info(f"Multiple linear regression returned the following parameters:")
    logging.info(f"v0: {v0}")
    logging.info(f"v1: {v1}")
    logging.info(f"v2: {v2}")
    costs4_per_iter_line_data = {'x': x, 'y': y}

    logging.info(f"Calling multiple linear regression with learning rate=0.3:")
    multi_linear_reg_info = multiple_linear_regression(scatter_data['x1'], scatter_data['x2'], scatter_data['y'], alpha_init=0.3)
    mlr_returns.append(multi_linear_reg_info)
    v0, v1, v2, costs5 = (
        multi_linear_reg_info['out']['v0'],
        multi_linear_reg_info['out']['v1'],
        multi_linear_reg_info['out']['v2'],
        multi_linear_reg_info['out']['costs']
    )
    x, y = zip(*list(enumerate(costs5, 1)))
    logging.info(f"Multiple linear regression returned the following parameters:")
    logging.info(f"v0: {v0}")
    logging.info(f"v1: {v1}")
    logging.info(f"v2: {v2}")
    costs5_per_iter_line_data = {'x': x, 'y': y}

    plt.figure()
    plt.plot(costs1_per_iter_line_data['x'], costs1_per_iter_line_data['y'], 'b-', label='alpha=0.001')
    plt.plot(costs2_per_iter_line_data['x'], costs2_per_iter_line_data['y'], 'r-', label='alpha=0.01')
    plt.plot(costs3_per_iter_line_data['x'], costs3_per_iter_line_data['y'], 'g-', label='alpha=0.1')
    plt.plot(costs4_per_iter_line_data['x'], costs4_per_iter_line_data['y'], 'y-', label='alpha=0.2')
    plt.plot(costs5_per_iter_line_data['x'], costs5_per_iter_line_data['y'], 'k-', label='alpha=0.3')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (J)')
    plt.legend(loc='upper right')
    logging.info(f"Writing cost over time for multiple linear regression graph to multiple_linear_regression_cost_per_iteration.png")
    plt.savefig('multiple_linear_regression_cost_per_iteration.png')

    return mlr_returns


def create_plot():
    slr_data = create_simple_linear_regression_plots()
    slr_data['out'].pop('costs')
    mlr_data = create_multiple_linear_regression_plots()
    [data['out'].pop('costs') for data in mlr_data]
    logging.info(f"-----------------------Review--------------------------------")
    logging.info(f"(Note: hiding costs for succinctness, scroll up for more     ")
    logging.info(f" details on what the specific costs were on each regression.)")
    logging.info(f"---------------Simple Linear Regression----------------------")
    logging.info(pp.pformat(slr_data))
    logging.info(f"--------------Multiple Linear Regression---------------------")
    logging.info(pp.pformat(mlr_data))


def main():
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    create_plot()


if __name__ == "__main__":
    main(sys.argv[1:])
