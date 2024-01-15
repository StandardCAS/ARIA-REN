import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('/root/NIGHT KOI DEMO 720.mov')

# Get the dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# Initialize a 3D numpy array
tablet = np.zeros((height, width, frames))
print(frames)
# Process each frame
# Initialize a list to store the points where the pixel value changes
changed_points = []


# Process each frame
for i in range(0,frames,1):
    print(i)
    ret, frame = cap.read()
    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if i > 0:
            # Compute the difference with the previous frame
            diff = cv2.absdiff(gray, prev_gray)
            # Update the tablet
            tablet[:, :, i] = diff
            # Find the points where the pixel value changes and add the frame number
            changed_points.extend([(x, y, i) for x, y in np.argwhere(diff != 0)])
        # Update the previous frame
        prev_gray = gray

print(len(changed_points))
# Save the tablet as a text file
np.save('nk720tablet.npy', tablet)

# Save the changed_points as a binary file
changed_points_npy = np.array(changed_points)
np.save('nk720cp.npy', changed_points)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d(changed_points, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = zip(*changed_points)

    ax.scatter(xs, ys, zs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig(filename)

# Call the function
#visualize_3d(changed_points, 'NIGHT KOI.png')




def save_to_obj(changed_points, filename):
    with open(filename, 'w') as f:
        vertex_count = 0
        # Iterate over the changed points
        for i in range(len(changed_points)):
            #print(changed_points[i])
            x,y,z = changed_points[i]
            #print(x,y,z)
            #print(vertex_count)
            # Write the vertices of the cube
            for dx, dy, dz in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]:
                f.write(f'v {x+dx} {y+dy} {z+dz}\n')
                vertex_count += 1
            # Write the faces of the cube
            f.write(f'f {vertex_count-7} {vertex_count-6} {vertex_count-2} {vertex_count-3}\n')
            f.write(f'f {vertex_count-4} {vertex_count-3} {vertex_count-2} {vertex_count-1}\n')
            f.write(f'f {vertex_count-7} {vertex_count-6} {vertex_count-4} {vertex_count-3}\n')
            f.write(f'f {vertex_count-8} {vertex_count-7} {vertex_count-3} {vertex_count-4}\n')
            f.write(f'f {vertex_count-6} {vertex_count-5} {vertex_count-1} {vertex_count-2}\n')
            f.write(f'f {vertex_count-8} {vertex_count-7} {vertex_count-5} {vertex_count-6}\n')
            if i%10000 == 0:
                print(vertex_count/(len(changed_points)*8))





def save_difference_video():
    import cv2
    import numpy as np

    # Assuming changed_points is a list of tuples where each tuple is (x, y, z)
    # and each of x, y, z are integers.

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

    # Sort the points by the z value (frame number)
    changed_points.sort(key=lambda x: x[2])

    # Initialize an empty frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    last_z = -1
    for point in changed_points:
        x, y, z = point
        # If we're at a new frame, write the last frame to the video file
        if z != last_z:
            out.write(frame)
            # Then create a new frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            last_z = z
        # Set the pixel at (y, x) to red
        frame[x, y] = [0, 0, abs(tablet[x,y,z])%255]  # BGR format

    # Write the last frame to the video file
    out.write(frame)

    # Release the VideoWriter
    out.release()


#save_difference_video()
