import os
import numpy as np
import matplotlib.pyplot as plt
logs = {}
for log in os.listdir('.'):
    a,b = log[4:-4].split('_')
    val = np.load(log)['val']
    logs.update({(int(a),int(b)):float(val)})

skeys = sorted(logs.keys())
vals = []
for key in skeys:
	print(key)
	print(logs[key])
	vals.append(logs[key])

plt.plot(range(len(vals)),vals)
plt.show()

