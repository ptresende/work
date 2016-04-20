#! /usr/bin/env python

import pdb

from sequence_generator import SeqGenerator
from units import PUnit


def main():
    Tw = 10000
    seq_gen = SeqGenerator()
    
    p = PUnit([1, 1, 1, 2])
    p2 = PUnit([1, 1, 2, 3])
    p3 = PUnit([1, 2, 4, 8])
    
    for t in range(Tw):
        (y, l) = seq_gen.step()
        
        p.new_symbol(y, l)
        p2.new_symbol(y, l)
        p3.new_symbol(y, l)

    p.draw()
    p2.draw()
    p3.draw()
    p.print_info()
    p2.print_info()
    p3.print_info()
    pdb.set_trace()

if __name__ == '__main__':
    main()
    