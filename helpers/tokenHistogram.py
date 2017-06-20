import numpy as np
import matplotlib.pyplot as plt
label_path = 'im2latex_formulas.norm.lst'
formulas = open(label_path).readlines()
data_path = 'im2latex_train_filter.lst'
tokenCounts = []
max_num_tokens = 0
with open(data_path) as fin:
    for line in fin:
        image_name, line_idx = line.strip().split()
        line_strip = formulas[int(line_idx)].strip()
        tokens = line_strip.split()
        tokenCounts.append(len(tokens))
        if len(tokens) > max_num_tokens:
            max_num_tokens = len(tokens)

print('max: ' + str(np.mean(max_num_tokens)))
print('mean: ' + str(np.mean(tokenCounts)))
print('median: ' + str(np.median(tokenCounts)))
histo, x = np.histogram(tokenCounts, bins=range(0,max_num_tokens+1))
plt.plot(range(0,max_num_tokens),histo)
plt.show()