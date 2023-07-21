import matplotlib
import matplotlib.pyplot as plt

import csv

# opening the CSV file
with open('data.csv', mode='r') as file:
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
plt.title('Data Plot')
plt.show()
