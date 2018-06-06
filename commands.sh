#!/bin/bash
for i in {0..1}
do
    python main.py mnist lb cooperative $i L0 40 1
    python main.py mnist lb cooperative $i L1 40 1
    python main.py mnist lb cooperative $i L2 40 1
done
exit 0
