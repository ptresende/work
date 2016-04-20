import sys

def internal_error():
    sys.exit("An unexpected internal error occurred.")



def unit_not_found():
    sys.exit("A unit that was assumed to be in a net was not found.")
