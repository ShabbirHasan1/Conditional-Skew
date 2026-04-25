""" Utility functions for NumPy """
import sys
import numpy as np

def true_ranges(bool_array):
    """
    Identifies ranges of consecutive True values in a 1D numpy array of booleans.
    Parameters:
    - bool_array (np.ndarray): A 1D numpy array of booleans.
    Returns:
    - np.ndarray: A 2D numpy array with two columns, where each row contains the start and end indices
      of consecutive True ranges in the input array.
    """
    # Initialize an empty list to store start and end indices of consecutive True ranges
    ranges = []
    # Initialize the start index of the current range of Trues to None
    start_index = None
    # Iterate over the array to find ranges of True values
    for i, value in enumerate(bool_array):
        if value:
            if start_index is None:
                start_index = i  # Mark the start of a new range of Trues
        else:
            if start_index is not None:
                # End of current range of Trues found; add the range to the list
                ranges.append([start_index, i])
                start_index = None  # Reset start_index for the next range of Trues
    # Check if the last element is True and hence a range ends with the last element
    if start_index is not None:
        ranges.append([start_index, len(bool_array)])
    # Convert the list of ranges to a 2D numpy array and return
    return np.array(ranges)

def func_blocks(x: np.ndarray, k: int, func, include_last=True):
    """
    Applies a specified function to blocks of length k in a 1-D array and returns the results.
    This function divides the input array `x` into blocks of size `k` and applies a given function `func`
    to each block. Blocks are formed column-wise in a temporary 2-D array before function application.
    Parameters:
    - x (np.ndarray): Input 1-dimensional numpy array.
    - k (int): Block size, indicating how many elements each block contains.
    - func (callable): A function to be applied to each block. Should accept a 1-D numpy array and return a single value.
    - include_last (bool): If True, includes the last block even if it's smaller than `k` (by padding with `np.nan`).
                           Defaults to True.
    Returns:
    - np.ndarray: A 1-D numpy array of the results obtained by applying `func` to each block.
                  Returns None if `k` is less than 1 or greater than the length of `x`.
    """
    if k < 1 or k > len(x):
        return None
    xmat = column_blocks(x, k, include_last)
    return np.apply_along_axis(func, 0, xmat)

