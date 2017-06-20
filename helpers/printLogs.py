import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
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

x = range(len(vals))
plt.plot(x, vals, color='blue')
x = np.expand_dims(np.array(x),1)
vals = np.expand_dims(np.array(vals),1)
regr = linear_model.LinearRegression()
regr.fit(x, vals)
plt.plot(x, regr.predict(x), color='red', linewidth=3)
plt.show()

