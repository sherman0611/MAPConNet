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

    pymesh.save_mesh('augmented_mesh.obj', rotated_mesh)
