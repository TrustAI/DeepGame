#!/bin/bash
for i in {0..0}
do
    python main.py cifar10 ub cooperative $i L1 40 0.2
done
exit 0
