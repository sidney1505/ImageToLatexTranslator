import os
import numpy as np
for batchfile in os.listdir('.'):
    batch = np.load(batchfile)
    images = batch['images']
    print(images.shape)