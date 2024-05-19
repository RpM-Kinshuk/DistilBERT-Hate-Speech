import matplotlib.pyplot as plt
import numpy as np

# Create some data
x = np.linspace(0, 10, 100)
y1 = x**2
y2 = x**3

# Plot the data
plt.plot(x, y1, label='y1')
plt.plot(x, y2, label='y2')

# Fill the area between the curves
plt.fill_between(x, y1, y2, alpha=0.5, color='blue')

# Add a legend
plt.legend()

# Show the plot
plt.show()