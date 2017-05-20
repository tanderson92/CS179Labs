# Plots cardiac twitch results for CS 179 Lab 6
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dt = .8/64.7 #ms
f=open("results.csv")
data = f.readlines()
floats = [None] * len(data)
for i, x in enumerate(data):
	try:
		floats[i] = float(x)
	except:
		print('FATAL:', x, 'could not be converted to float...')
		quit()
time = [dt * i for i in range(len(data))]
plt.plot(time, floats)
plt.title("Percent Cardiac Tissue Activation vs Time")
plt.xlabel('Time (ms)')
plt.ylabel('Percent of RU')
plt.savefig('plot.png')
