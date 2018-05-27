#!/bin/bash
for i in {1..1}
do
    python main.py cifar10 ub competitive $i L1 40 0.2
done
exit 0
