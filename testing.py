import os

pid = 0
affinity = os.sched_getaffinity(pid)
print(affinity)
