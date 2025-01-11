
"""
Integration test for the main function in your_module.

This test captures the standard output of the main function and asserts
that it matches the expected output.

Functions:
    test_main_output(capsys): Tests the output of the main function.
"""
from package import main  # Replace 'your_module' with the actual module name

def test_main_output(capsys):
    """ Tests the output of the main function. """
    main.main()
    captured = capsys.readouterr()
    assert captured.out == "Hello, World!\n"
