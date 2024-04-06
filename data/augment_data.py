import pymesh

def random_augment():
    augmented_mesh = pymesh.load_mesh('id0_0.obj')
    pymesh.save_mesh('augmented_mesh.obj', augmented_mesh)

random_augment()