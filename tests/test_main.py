# -*- coding: utf-8 -*-

import pytest
import statistics
from unittest.mock import patch, MagicMock, call
from univariate_linear_regression.main import *

__author__ = "Jesse Moore"
__copyright__ = "Jesse Moore"
__license__ = "mit"


def test_make_correlated_sequence():
    x = [ i for i in range(1, 101) ]
    y = make_correlated_sequence(x, random_array_of_size(100))
    #calulate variance
    #1. get means
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    #2. subtract mean from arrays
    a = [ a - mean_x for a in x ]
    b = [ b - mean_y for b in y ]
    #3 calculate A*B. A^2, B^2, for each value
    ab = [ i * j for i, j in zip(a, b) ]
    aa = [ i**2 for i in a ]
    bb = [ j**2 for j in b ]
    #4 get sums
    sum_ab = sum(ab)
    sum_aa = sum(aa)
    sum_bb = sum(bb)
    #5 get correlation
    correlation = sum_ab / math.sqrt(sum_aa * sum_bb)
    assert round(correlation, 3) >= .9


@patch('univariate_linear_regression.main.create_plot')
def test_main(mock_plot):
     main()
     assert mock_plot.called


#def test_cost_function():
#    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#    bias = 0
#    weight = 1
#    assert cost_function(x, y, bias, weight) == 0
#    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#    y = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#    bias = 0
#    weight = 1
#    assert cost_function(x, y, bias, weight) == 200


def test_create_best_fit_bias_and_weight():
    data = {
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    assert (0, 1) == create_best_fit_bias_and_weight(data)
    data = {
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'y': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    }
    assert (2, 1) == create_best_fit_bias_and_weight(data)
    data = {
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'y': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    }
    assert (0, 2) == create_best_fit_bias_and_weight(data)


def test_create_line():
    line = list(zip(range(1, 11), [ 2+2*x for x in range(1, 11)]))
    line_points = create_line(range(1, 11), 2, 2)
    assert line == line_points


@patch('univariate_linear_regression.main.plt.plot')
@patch('univariate_linear_regression.main.plt.savefig')
def test_create_plot(mock_savefig, mock_plot):
    data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
    create_plot()
    # assert mock_plot.call_args_list == [
    #     call([1, 2, 3], [4, 5, 6], '.'),
    #     call((1, 2, 3), (4.0, 5.0, 6.0), 'g-'),
    #     #call((1, 2, 3), (4.0, 5.0, 6.0), 'b-')
    #     #call((1, 2, 3), (21.25, 28.0, 34.75), 'b-')
    # ]
    assert mock_savefig.called_with('/app/plot.png')



def test_random_array_of_size():
    arr = random_array_of_size(100)
    assert len(arr) == 100
    for elem in arr:
        assert elem >= 1 and elem <= 100
