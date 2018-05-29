#!/bin/bash
for i in {0..0}
do
    python main.py cifar10 ub cooperative $i L1 40 0.5
    python main.py cifar10 ub competitive $i L1 40 0.5
done
exit 0
