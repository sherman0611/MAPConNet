import numpy as np
import pymesh

def random_augment():
    original_mesh = pymesh.load_mesh('id0_0.obj')

    # generate random rotation angles for each axis
    angles = np.random.uniform(low=0, high=2*np.pi, size=3)

    # perform the rotation
    augmented_mesh = pymesh.rotate(original_mesh, angles)

    pymesh.save_mesh('augmented_mesh.obj', augmented_mesh)

random_augment()