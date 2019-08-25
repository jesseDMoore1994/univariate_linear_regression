# -*- coding: utf-8 -*-

import pytest
from unittest.mock import patch
from univariate_linear_regression.main import *

__author__ = "Jesse Moore"
__copyright__ = "Jesse Moore"
__license__ = "mit"


@patch('univariate_linear_regression.main.create_plot')
def test_main(mock_plot):
     main()
     assert mock_plot.call_args_list[0][0][0]['x']
     assert mock_plot.call_args_list[0][0][0]['y']


def test_random_array_of_size():
    arr = random_array_of_size(10)
    assert len(arr) == 10
    for elem in arr:
        assert elem >= 1 and elem <= 25
