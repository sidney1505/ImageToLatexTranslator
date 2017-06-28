import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
logs = {}
for log in os.listdir('.'):
    a,b = log[4:-4].split('_')
    log = np.load(log)
    logs.update({(int(a),int(b)):(float(log['train_loss']),float(log['val_loss']), \
    	float(log['train_accuracy']),float(log['val_accuracy']))})

skeys = sorted(logs.keys())
train_loss = []
val_loss = []
train_accuracy = []
val_accuracy = []
for key in skeys:
	print(key)
	print(logs[key])
	print(logs[key][0])
	train_loss.append(logs[key][0])
	print(logs[key][1])
	val_loss.append(logs[key][1])
	print(logs[key][2])
	train_accuracy.append(logs[key][2])
	print(logs[key][3])
	val_accuracy.append(logs[key][3])

plt.plot(train_loss[10:], color='blue')
plt.plot(val_loss[10:], color='red')
plt.show()
plt.plot(train_accuracy[10:], color='blue')
plt.plot(val_accuracy[10:], color='red')
plt.show()
#x = range(len(vals))
#x = np.expand_dims(np.array(x),1)
#vals = np.expand_dims(np.array(vals),1)
#regr = linear_model.LinearRegression()
#regr.fit(x, vals)
#plt.plot(x, regr.predict(x), color='red', linewidth=3)
