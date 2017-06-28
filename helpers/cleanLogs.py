import os
import shutil
for file in os.listdir('.'):
	logs = []
	path = file + '/logs'
	if os.path.exists(path):
		logs = os.listdir(path)
	if len(logs) < 50:
		shutil.rmtree(file, ignore_errors=True)

