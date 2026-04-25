""" Python utility functions """
import time

def print_vec(x, fmt:str = "%10.4f", delim=" ", title=None, with_num:bool=False,
    num0=0, fmt_num="%10d", trailer=None, labels=None, fmt_label="%10s"):
    """ print vector with given format and delimiter """
    if title:
        print(title, end="")
    if with_num: # print numeric labels
        print(delim.join(fmt_num%(i+num0) for i in range(len(x))))
    if labels:
        print(delim.join(fmt_label%label for label in labels))
    print(delim.join(fmt%y for y in x))
    if trailer:
        print(trailer, end="")
