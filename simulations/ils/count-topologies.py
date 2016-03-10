#!/bin/env python

import re
import sys

break_points = [float(arg) for arg in sys.argv[1:]] + [100000000]
def get_bin(time_point):
	for i in xrange(len(break_points)-1):
		if break_points[i] <= time_point < break_points[i+1]:
			return i
	return i


rex = re.compile(r"\[(?P<length>\d*)\]\((?P<outgroup>\d):(?P<tau2>\d\.\d*),\(\d:(?P<tau1>\d.\d*),\d:\d.\d*\).*\);")

counts = dict()
for line in sys.stdin:
	match = rex.match(line)
	L = int(match.group("length"))
	bin1 = get_bin(float(match.group("tau1")))
	bin2 = get_bin(float(match.group("tau2")))
	topology = (match.group("outgroup"), bin1, bin2)
	if topology not in counts:
		counts[topology] = 0
	counts[topology] += L

n = sum(counts.values())
for topology in counts:
	print topology, float(counts[topology]) / n
