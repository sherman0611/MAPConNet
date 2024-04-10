import os
import pymesh
import h5py
import cv2
import numpy as np
from glob import glob
from util.util import visualise_geometries

def view_meshes(ids, poses):
    save_dir = 'vis/npt-data'
    os.makedirs(save_dir, exist_ok=True)
    points = []
    faces = []
    names = []
    # indices = np.random.randint(6890, size=69)
    for id in ids:
        for pose in poses:
            mesh = pymesh.load_mesh(f'../data/npt-data/id{id}_{pose}.obj')
            points.append(mesh.vertices)
            faces.append(mesh.faces)
            names.append(f'{id}_{pose}')
            import pdb; pdb.set_trace()
    visualise_geometries(points, names=names, save_dir=save_dir) #faces=faces,

def view_mg():
    data_dir = '../data/Multi-Garment_dataset/'
    sort_key = lambda p: int(os.path.basename(p))
    datapaths = sorted(glob(os.path.join(data_dir, '*')), key=sort_key)
    for path in datapaths:
        mesh = pymesh.load_mesh(os.path.join(path, 'smpl_registered.obj'))
        visualise_geometries([mesh.vertices], faces=[mesh.faces])
        import pdb; pdb.set_trace()

def view_dfaust(gender='m'):
    dfaust_dir = '../data/DFAUST'
    f = h5py.File(os.path.join(dfaust_dir, f'registrations_{gender}.hdf5'), 'r')
    ks = []
    out_dict = {}
    for k in f.keys():
        if k != 'faces':
            ks.append(k)
            identity = k[:5]
            pose = k[6:]
            if identity not in out_dict:
                out_dict[identity] = [pose]
            elif pose not in out_dict[identity]:
                out_dict[identity].append(pose)
    import pdb; pdb.set_trace()
    #     print(k)
    #     sequence = f[k].value.transpose([2, 0, 1])
    #     for i in range(10):
    #         visualise_geometries(sequence[10*i:10*i+10,], faces=[f['faces'].value]*10)
    # verts = f[sidseq].value.transpose([2, 0, 1])
    # faces = f['faces'].value

def view_dfaust2(freq=10, sids=None, sequences=None, num=10, save=False):
    if type(freq).__name__ == 'list':
        print(f'Frames: {freq}')
    else:
        print(f'Frequency: {freq}')
    dfaust_dir = '../data/DFAUST'
    if save:
        save_dir = './vis/DFAUST'
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None
    if sids is None:
        sids = ['50002', '50004', '50007', '50009', '50020',
                '50021', '50022', '50025', '50026', '50027']
    for sid in sids:
        if sequences is None:
            seq_key = lambda p: os.path.basename(p)[0]
            seqs = sorted(glob(os.path.join(dfaust_dir, sid, '*')), key=seq_key)
        else:
            seqs = [os.path.join(dfaust_dir, sid, seq) for seq in sequences]
        for seq in seqs:
            if not os.path.exists(seq):
                print(f'{seq} does not exist')
                continue
            print(f'{sid}: {os.path.basename(seq)}')
            obj_key = lambda p: int(os.path.basename(p).split('_')[-1][:-4])
            objs = sorted(glob(os.path.join(seq, '*.obj')), key=obj_key)
            points = []
            faces = []
            names = []
            for i, obj in enumerate(objs):
                if ((type(freq).__name__ == 'list') and ((i in freq) or (i-len(objs) in freq))) or ((type(freq).__name__ == 'int') and (i % freq == 0)):
                    mesh = pymesh.load_mesh(obj)
                    points.append(mesh.vertices)
                    faces.append(mesh.faces)
                    names.append(os.path.basename(obj)[:-4])
                if len(points) == num or (i+1 == len(objs) and len(points) > 0):
                    print(f'Last frame: {i}')
                    visualise_geometries(points, faces=faces, save_dir=save_dir, names=names)
                    if save:
                        import pdb; pdb.set_trace()
                    points = []
                    faces = []

def write_dfaust():
    def write_mesh_as_obj(fname, verts, faces):
        with open(fname, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    dfaust_dir = '../data/DFAUST'
    males = os.path.join(dfaust_dir, f'registrations_m.hdf5')
    females = os.path.join(dfaust_dir, f'registrations_f.hdf5')
    for path in [males, females]:
        with h5py.File(path, 'r') as f:
            for k in f.keys():
                if k!= 'faces':
                    sid_dir = os.path.join(dfaust_dir, k[:5])
                    os.makedirs(sid_dir, exist_ok=True)
                    seq_dir = os.path.join(sid_dir, k[6:])
                    os.makedirs(seq_dir, exist_ok=True)
                    sequence = f[k].value.transpose([2, 0, 1])
                    for i, verts in enumerate(sequence):
                        fpath = os.path.join(seq_dir, f'{k[:5]}_{k[6:]}_{i}.obj')
                        write_mesh_as_obj(fpath, verts, f['faces'].value)
                    print(f'{k}: {i+1} meshes')

def dfaust_list(save_freq=1):
    dfaust_dir = '../data/DFAUST'
    sids = ['50002', '50004', '50007', '50009', '50020',
            '50021', '50022', '50025', '50026', '50027']
    save_count = 0
    save_path = f'dfaust_list_freq{save_freq}.txt'
    with open(save_path, 'w') as f:
        for sid in sids:
            seq_key = lambda p: os.path.basename(p)[0]
            seqs = sorted(glob(os.path.join(dfaust_dir, sid, '*')), key=seq_key)
            for seq in seqs:
                obj_key = lambda p: int(os.path.basename(p).split('_')[-1][:-4])
                objs = sorted(glob(os.path.join(seq, '*.obj')), key=obj_key)
                for i, obj in enumerate(objs):
                    if i % save_freq == 0:
                        f.write(os.path.basename(obj) + '\n')
                        save_count += 1
                        if save_count % 100 == 0:
                            print(f'{save_count} examples saved')
    print(f'{save_count} examples in total.')

def dfaust_list2(save_freq=5):
    dfaust_dir = '../data/DFAUST'
    sids = ['50002', '50004', '50007', '50009', '50020',
            '50021', '50022', '50025', '50026', '50027']
    sids_pos = ['50004', '50021', '50025', '50022', '50020']
    sids_neg = ['50007', '50002', '50027', '50026', '50009']
    seqs = ['chicken_wings', 'hips', 'jiggle_on_toes', 'jumping_jacks', 'knees', 'light_hopping_loose', 'light_hopping_stiff',
            'one_leg_jump', 'one_leg_loose', 'punching', 'running_on_spot', 'shake_arms', 'shake_hips', 'shake_shoulders']

    np.random.seed(12345)
    np.random.shuffle(sids_pos)
    np.random.shuffle(sids_neg)
    np.random.shuffle(seqs)

    sids_train = sids_pos[:3] + sids_neg[:3]
    sids_test = sids_pos[3:] + sids_neg[3:]
    seqs_train = seqs[:9]
    seqs_test = seqs[9:]

    save_count = 0
    for split in ['train', 'test']:
        print(f'Processing split: {split}.')
        save_path = f'dfaust_list2_freq{save_freq}_{split}.txt'
        if split == 'train':
            sids = sids_train
            seqs = seqs_train
        elif split == 'test':
            sids = sids_test
            seqs = seqs_test
        with open(save_path, 'w') as f:
            for sid in sids:
                print(f'\tProcessing subject: {sid}.')
                seqs_existing = glob(os.path.join(dfaust_dir, sid, '*'))
                for seq in seqs_existing:
                    if os.path.basename(seq) not in seqs:
                        continue
                    obj_key = lambda p: int(os.path.basename(p).split('_')[-1][:-4])
                    objs = sorted(glob(os.path.join(seq, '*.obj')), key=obj_key)
                    N = len(objs)
                    fids = range(int(0.1*N), int(0.9*N))[::save_freq]
                    print(f'\t\tProcessing sequence: {os.path.basename(seq)}, {len(fids)} poses.')
                    for i in fids:
                        f.write(os.path.basename(objs[i]) + '\n')
                        save_count += 1
    print(f'{save_count} examples in total.')

def dfaust_list3():
    objs = glob('../data/dfaust-data/train/*.obj')
    out_file = 'dfaust_list_train.txt'
    identities = []
    for obj in objs:
        fname = os.path.basename(obj)
        identity = fname.split('_')[0][2:]
        if identity not in identities:
            identities.append(identity)

    with open(out_file, 'w') as file:
        file.write(','.join(identities) + '\n')
        for obj in objs:
            file.write(os.path.basename(obj) + '\n')

def data_list(percentage, num_total, mode):
    datapath_labelled = []
    datapath_unlabelled = []
    if mode == 'human':
        ids_all = np.arange(0, 16)
        poses_all = np.arange(200, 600)
    elif mode == 'animal':
        ids_all = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 23, 24, 26, 27, 28, 29, 30, 31, 34, 35, 38, 39])
        poses_all = np.arange(0, 400)
    else:
        raise Exception

    np.random.seed(1234)
    num_id = round(len(ids_all) * percentage / 100)
    num_pose = round(len(poses_all) * percentage / 100)
    ids = np.random.choice(ids_all, size=num_id, replace=False)
    poses = np.random.choice(poses_all, size=num_pose, replace=False)
    unused_ids = [i for i in ids_all if i not in ids]
    unused_poses = [i for i in poses_all if i not in poses]

    if len(ids) > 0 and len(poses) > 0:
        for _ in range(num_total):
            identity_i = np.random.choice(ids, replace=True)
            identity_p = np.random.choice(poses, replace=True)
            datapath_labelled.append([identity_i, identity_p])
        fname = f'datapath_labelled_{percentage}%.txt'
        if mode == 'animal':
            fname = 'animal_' + fname
        with open(fname, 'w') as f:
            f.write(','.join([str(i) for i in ids]) + '\n')
            f.write(','.join([str(p) for p in poses]) + '\n')
            for i, p in datapath_labelled:
                f.write(f'{i},{p}\n')

    if len(unused_ids) > 0 and len(unused_poses) > 0:
        for _ in range(num_total):
            identity_i = np.random.choice(unused_ids, replace=True)
            identity_p = np.random.choice(unused_poses, replace=True)
            datapath_unlabelled.append([identity_i, identity_p])
        fname = f'datapath_unlabelled_{100 - percentage}%.txt'
        if mode == 'animal':
            fname = 'animal_' + fname
        with open(fname, 'w') as f:
            f.write(','.join([str(i) for i in unused_ids]) + '\n')
            f.write(','.join([str(p) for p in unused_poses]) + '\n')
            for i, p in datapath_unlabelled:
                f.write(f'{i},{p}\n')

def view_faust():
    from plyfile import PlyData
    sort_key = lambda p: int(p[:-4].split('_')[-1])
    paths = sorted(glob('../data/MPI-FAUST/training/registrations/*ply'), key=sort_key)
    for p in paths:
        mesh = PlyData.read(p)
        points = np.asarray(mesh.elements[0].data.tolist())
        faces = np.asarray(mesh.elements[1].data.tolist()).squeeze()
        visualise_geometries([points], [faces])
        # import pdb; pdb.set_trace()

def crop_images_in_dir(img_dir, top, bottom):
    img_dir = os.path.join('vis', 'qualitative examples', img_dir)
    out_dir = img_dir + '_cropped'
    os.makedirs(out_dir, exist_ok=True)
    fs = glob(os.path.join(img_dir, '*.jpg'))
    for f in fs:
        img = cv2.imread(f)
        img_new = img[top:bottom]
        cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), img_new)
        # import pdb; pdb.set_trace()

if __name__ == '__main__':
    # sids = ['50002', '50004', '50007', '50009', '50020',
    #         '50021', '50022', '50025', '50026', '50027']
    # seqs = ['chicken_wings', 'hips', 'jiggle_on_toes', 'jumping_jacks', 'knees', 'light_hopping_loose', 'light_hopping_stiff',
    #         'one_leg_jump', 'one_leg_loose', 'punching', 'running_on_spot', 'shake_arms', 'shake_hips', 'shake_shoulders']
    # view_meshes([2, 3, 5], [601, 602, 603, 604, 605])
    # view_meshes([5,], [601,])
    # view_mg()
    # view_dfaust('f')
    # write_dfaust()
    # dfaust_list(save_freq=11)
    # view_dfaust2(freq=20, sids=['50007'], sequences=['jiggle_on_toes'], num=5, save=True) #, sequences=seqs)

    # data_list(50, 4000, 'human')
    # data_list(0, 11600, 'animal')
    # view_faust()

    # dfaust_list2()
    # dfaust_list3()

    # crop_images_in_dir('245_toy_37_474', 250, 750)
    # crop_images_in_dir('117_id24_740', 100, 900)
    # crop_images_in_dir('157_groundTruth_157', 100, 900)
    # crop_images_in_dir('mg38_12_groundTruth_012', 100, 900)
    # crop_images_in_dir('3_id24_638', 100, 900)
    # crop_images_in_dir('4_id26_649', 100, 900)
    crop_images_in_dir('52_id27_699', 100, 900)

