
"""
This module contains a unit test for the function dummy_function.

The dummy_function is a simple function that adds two numbers together.

Example usage:
    result = dummy_function(1, 2)
    print(result)  # Output: 3
"""

import unittest


def dummy_function(a, b):
    """
    This is a dummy function that adds two numbers.
    """
    return a + b


class TestDummyFunction(unittest.TestCase):
    """
    This is a test class for the dummy_function.
    """

    def test_dummy_function(self):
        """
        This is a test method for the dummy_function.
        """
        self.assertEqual(dummy_function(1, 2), 3)
