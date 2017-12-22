#execute with python 3 interpreter

import numpy as np

print("###################################\n\
# scenario: NC\n\
# net: mid-caffenet\n\
# strategy: icarl\n\
###################################")
print("RunID	Batch0	Batch1	Batch2	Batch3	Batch4	Batch5	Batch6	Batch7	Batch8")

results = np.zeros((3, 9), np.float32)
imgs = [10, 25, 100]


r = 0
for run in range(3):
    filename = "results/test_run" + str(run) + "._lr1_0.02_lro_0.05_" + '' + "vgg.txt"
    file = open(filename)
    increment = -1
    for line in file:
        if line.startswith("Increment"):
            increment += 1
        if line.startswith("iCaRL top 1 accuracy:"):
            results[r][increment] = float(line.split(": ")[1])
    r += 1

for run in range(3):
    print(run, end='')
    for increment in range(9):
        print("\t{:.2%}".format(results[run][increment]), end='')
    print("\n", end='')

avgs = results.mean(axis=0)
print("avg", end='')
for increment in range(9):
    print("\t{:.2%}".format(avgs[increment]), end='')
print("\n", end='')

stds = results.std(axis=0)
print("dev.std", end='')
for increment in range(9):
    print("\t{:.2%}".format(stds[increment]), end='')
print("\n", end='')
