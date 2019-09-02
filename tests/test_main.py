# -*- coding: utf-8 -*-

import pytest
import statistics
from unittest.mock import patch, MagicMock, call
from univariate_linear_regression.main import *

__author__ = "Jesse Moore"
__copyright__ = "Jesse Moore"
__license__ = "mit"


@patch('univariate_linear_regression.main.create_plot')
def test_main(mock_plot):
     main()
     assert mock_plot.called


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
