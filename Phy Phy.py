import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from concurrent.futures import ThreadPoolExecutor

class PhysicsSimulator:
    def __init__(self, num_objects, size=5000):
        # Initialize the positions and velocities of the objects
        self.positions = np.random.rand(num_objects, 3) * size  # Random positions within the specified size
        self.velocities = np.zeros((num_objects, 3))  # Initial velocities are zero

        # Create a figure and an axis for the animation
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Set the limits of the axes
        self.ax.set_xlim([0, size])
        self.ax.set_ylim([0, size])
        self.ax.set_zlim([0, size])

        # Create a scatter plot object for the animation
        self.scatter = self.ax.scatter(*self.positions.T)

    def check_collision(self, i, j):
        if np.linalg.norm(self.positions[i] - self.positions[j]) < 1:  # If the particles are close enough to collide
            # Calculate the direction of the tangent at the point of contact
            tangent = np.cross(self.positions[i] - self.positions[j], [0, 0, 1])
            tangent /= np.linalg.norm(tangent)

            # Update the velocities of the particles
            self.velocities[i] = 0.2 * np.dot(self.velocities[i], tangent) * tangent
            self.velocities[j] = 0.2 * np.dot(self.velocities[j], tangent) * tangent

    def update(self, i):
        # Update the positions based on the velocities
        self.positions += self.velocities

        # Apply gravity
        self.velocities[:, 2] -= 9.8  # Decrease z-velocity to simulate gravity

        # Check for collisions with the ground and reverse z-velocity if a collision occurred
        collisions = self.positions[:, 2] < 0
        self.velocities[collisions, 2] *= -0.2

        # Check for collisions between particles
        with ThreadPoolExecutor(max_workers=200) as executor:
            for i in range(len(self.positions)):
                for j in range(i+1, len(self.positions)):
                    executor.submit(self.check_collision, i, j)

        # Update the scatter plot object
        self.scatter._offsets3d = self.positions.T


# Create a simulator with 500 objects in a 5000x5000x5000 field
simulator = PhysicsSimulator(500, size=50)

# Create the animation
ani = FuncAnimation(simulator.fig, simulator.update, frames=100, interval=100)

plt.show()
