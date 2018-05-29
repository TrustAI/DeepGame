#!/bin/bash
for i in {10..18}
do
    python main.py cifar10 ub cooperative $i L1 40 0.2
    python main.py cifar10 ub competitive $i L1 40 0.2
done
exit 0
