#!/bin/bash
python -m cProfile -o _profile_stats $1
gprof2dot -f pstats _profile_stats -o _call_graph.dot
xdot _call_graph.dot