import matplotlib
import matplotlib.pyplot as plt

import csv

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.xlim(-4, 10)
plt.ylim(0, 255)
# opening the CSV file
with open('points.csv', mode='r') as file:
    csvFile = csv.reader(file)

    x = []
    y = []

    for lines in csvFile:
        x.append(float(lines[0]))
        y.append(float(lines[1]))

plt.scatter(x, y, marker='o', facecolors='none', edgecolors='b')
# plt.plot(x, y)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.title('Points Plot')

plt.subplot(1, 2, 2)
plt.xlim(-4, 10)
plt.ylim(0, 255)
with open('bins.csv', mode='r') as file:
    csvFile = csv.reader(file)

    x = []
    y = []

    for lines in csvFile:
        x.append(float(lines[0]))
        y.append(float(lines[1]))

plt.scatter(x, y, marker='o', facecolors='none', edgecolors='b')
# plt.plot(x, y)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.title('Bins Plot')

plt.tight_layout()
plt.show()
