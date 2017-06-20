import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
vals = range(10)
vals[4] = vals[4] + 2
vals[7] = vals[7] - 1
vals[2] = vals[2] - 1
x = range(len(vals))
plt.plot(x, vals, color='blue')
x = np.expand_dims(np.array(x),1)
vals = np.expand_dims(np.array(vals),1)
regr = linear_model.LinearRegression()
regr.fit(x, vals)
plt.plot(x, regr.predict(x), color='red', linewidth=3)
plt.show()

