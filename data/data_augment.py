import numpy as np
import pymesh

def random_rotate(mesh):
    # generate random rotation angles for each axis
    angles = np.random.uniform(low=0, high=2*np.pi, size=3)

    # perform the rotation using rotation matrices
    rotation_matrices = [pymesh.Quaternion.fromAxisAngle(axis, angle).to_matrix()
                         for axis, angle in zip(np.eye(3), angles)]
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = np.dot(rotation_matrices[2], np.dot(rotation_matrices[1], rotation_matrices[0]))

    # apply rotation to vertices
    rotated_vertices = np.dot(mesh.vertices, rotation_matrix[:3, :3].T)

    rotated_mesh = pymesh.form_mesh(rotated_vertices, mesh.faces)
    
    return rotated_mesh

def random_scale(mesh, min_scale=0.8, max_scale=1.2):
    # generate random scaling factors for each axis
    scales = np.random.uniform(low=min_scale, high=max_scale, size=3)

    # apply scaling to vertices
    scaled_vertices = mesh.vertices * scales.reshape(1, -1)

    scaled_mesh = pymesh.form_mesh(scaled_vertices, mesh.faces)

    return scaled_mesh

def uniform_noise_corruption(mesh, noise_scale=0.01):
    # generate random noise
    noise = np.random.uniform(low=-noise_scale, high=noise_scale, size=mesh.vertices.shape)

    # add noise to vertices
    corrupted_vertices = mesh.vertices + noise

    corrupted_mesh = pymesh.form_mesh(corrupted_vertices, mesh.faces)

    return corrupted_mesh

# augment mesh using all transformations
def random_augmentations(mesh, min_scale=0.8, max_scale=1.2, noise_scale=0.01):
    augmented_mesh = random_rotate(mesh)
    augmented_mesh = random_scale(augmented_mesh, min_scale, max_scale)
    augmented_mesh = uniform_noise_corruption(augmented_mesh, noise_scale)

    pymesh.save_mesh('augmented_mesh.obj', augmented_mesh)

    return augmented_mesh