#!/bin/bash
for i in {0..10}
do
    python main.py mnist ub cooperative $i L2 10 1
    python main.py mnist lb cooperative $i L2 0.01 1
done
exit 0
