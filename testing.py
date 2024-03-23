# Import  libraries
import os
import sys
import torch
from torch import nn
import torch
import time

# Python program to explain os.sched_setaffinity() method

# importing os module
import os

# Get the number of CPUs
# in the system
# using os.cpu_count() method
print("Number of CPUs:", os.cpu_count())

# Get the set of CPUs
# on which the calling process
# is eligible to run. using
# os.sched_getaffinity() method
# 0 as PID represents the
# calling process
pid = 0
affinity = os.sched_getaffinity(pid)

# Print the result
print("Process is eligible to run on:", affinity)

# Change the CPU affinity mask
# of the calling process
# using os.sched_setaffinity() method

# Below CPU affinity mask will
# restrict a process to only
# these 2 CPUs (0, 1) i.e process can
# run on these CPUs only
affinity_mask = {6, 7, 8, 9, 10, 11}
pid = 0
os.sched_setaffinity(0, affinity_mask)
print("CPU affinity mask is modified for process id % s" % pid)

# Now again, Get the set of CPUs
# on which the calling process
# is eligible to run.
pid = 0
affinity = os.sched_getaffinity(pid)

# Print the result
print("Now, process is eligible to run on:", affinity)

def initialize_parameters() :
    e = torch.rand([300, 49512])
    w = torch.rand([4, 256, 300])
    u = torch.rand([4, 256, 256])
    v = torch.rand([4, 300, 256])
    hidden = torch.rand([4, 4096, 256])
    input = torch.rand([4096, 4, 49512])
    return e, w, u, v, hidden, input

def add(val1, val2):
    val3 = torch.add(val1, val2)
    return val3

def embedding_convert(input_vector, embedding):
    output = torch.matmul(input_vector, torch.t(embedding))
    return output

def rnn_cell_computation(input_vector, hidden_state, weight, u, v, embedding):
    h1 = torch.matmul(input_vector, torch.t(weight))
    h2 = torch.matmul(hidden_state, torch.t(u))
    h = torch.add(h1, h2)
    out = torch.matmul(h, torch.t(v))
    return out, h

def output_one_hot(output, embedding) :
    out = torch.matmul(output, embedding)
    out = torch.softmax(out,  dim=1)
    return out

start = time.time()
stack_length = 4
sequence_length = 4
device = "cpu"
hidden_size = 256
mini_batch_size = 512

output_vector = [[None for i in range(sequence_length)] for j in range(stack_length)]
hidden_vector = [[None for i in range(sequence_length)] for j in range(stack_length)]

e, w, u, v, h, input = initialize_parameters()
input_vector = embedding_convert(input, e)


hidden_state = h[0, :, :]
for i in range(sequence_length) :
    output_vector[0][i], hidden_state = rnn_cell_computation(input_vector = input_vector[:, i, :], hidden_state= hidden_state, weight= w[0, :, :], u = u[0, :, :],v = v[0, :, :], embedding= e)

for i in range(1, stack_length) :
    hidden_state = h[i, :, :]
    for j in range(sequence_length) :
        output_vector[i][j], hidden_state = rnn_cell_computation(input_vector = output_vector[i-1][j], hidden_state= hidden_state, weight= w[i, :, :], u = u[i, :, :],v = v[i, :, :], embedding= e)

for i in range(sequence_length) :
    output_vector[-1][i] = output_one_hot(output_vector[-1][i], e)

end = time.time()
print("Total Time : ", end - start)
print("Done")


