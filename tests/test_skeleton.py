# -*- coding: utf-8 -*-

import pytest
from univariate_linear_regression.skeleton import fib

__author__ = "Jesse Moore"
__copyright__ = "Jesse Moore"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
