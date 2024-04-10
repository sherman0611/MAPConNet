import torch
import numpy as np
import os
import sys
import importlib

def feature_normalize(feature_in):
    feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature_in_norm = torch.div(feature_in, feature_in_norm)
    return feature_in_norm

def weighted_l1_loss(input, target, weights):
    out = torch.abs(input - target)
    out = out * weights.expand_as(out)
    loss = out.mean()
    return loss

def mse_loss(input, target=0):
    return torch.mean((input - target)**2)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls

def print_network(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).'
            % (type(model).__name__, num_params / 1000000))

def save_network(net, label, epoch, opt, _use_new_zipfile_serialization=False):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, save_filename)
    torch.save(net.cpu().state_dict(), save_path, _use_new_zipfile_serialization=_use_new_zipfile_serialization)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()

def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, save_filename)
    if not os.path.exists(save_path):
        print('not find model :' + save_path + ', do not load model!')
        return net
    weights = torch.load(save_path)
    try:
        net.load_state_dict(weights)
    except KeyError:
        print('key error, not load!')
    except RuntimeError as err:
        print(err)
        net.load_state_dict(weights, strict=False)
        print('loaded with strict=False')
    return net

def print_current_errors(opt, epoch, i, errors, t):
    message = 'epoch: %d, iters: %d, time: %.3f, ' % (epoch, i, t)
    for k, v in errors.items():
        if k != 'lr':
            v = v.mean().float()
            message += '%s: %.3f, ' % (k, v)
        else:
            message += '%s: %f' % (k, v)

    print(message)
    log_name = os.path.join(opt.exp_dir, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)

def init_regul(source_vertices, source_faces):
    sommet_A_source = source_vertices[source_faces[:, 0]]
    sommet_B_source = source_vertices[source_faces[:, 1]]
    sommet_C_source = source_vertices[source_faces[:, 2]]
    target = []
    target.append(np.sqrt( np.sum((sommet_A_source - sommet_B_source) ** 2, axis=1)))
    target.append(np.sqrt( np.sum((sommet_B_source - sommet_C_source) ** 2, axis=1)))
    target.append(np.sqrt( np.sum((sommet_A_source - sommet_C_source) ** 2, axis=1)))
    return target

def get_target(vertice, face, size):
    target = init_regul(vertice,face)
    target = np.array(target)
    target = torch.from_numpy(target).float() #.cuda()
    #target = target+0.0001
    target = target.unsqueeze(1).expand(3,size,-1)
    return target

def compute_score(points, faces, target):
    target = target.to(points.device)
    score = 0
    sommet_A = points[:,faces[:, 0]]
    sommet_B = points[:,faces[:, 1]]
    sommet_C = points[:,faces[:, 2]]

    score = torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_B) ** 2, dim=2)) / target[0] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_B - sommet_C) ** 2, dim=2)) / target[1] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_C) ** 2, dim=2)) / target[2] -1)
    return torch.mean(score)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 3)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
    return feat_mean, feat_std

def visualise_geometries(meshes, faces=None, colors=None, names=None, save=False, save_dir=None, dataset_mode='human', angle=None, paper_order=False, zooms=None):
    import open3d as o3d

    if dataset_mode == 'human':
        gap = 1.3
        rot_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    else:
        gap = 2.5
        rot_mat = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    if angle is not None:
        angle_ = angle * np.pi / 180
        rot_mat_new = np.array([[np.cos(angle_), 0, np.sin(angle_)], [0, 1, 0], [-np.sin(angle_), 0, np.cos(angle_)]])
        rot_mat = np.dot(rot_mat_new, rot_mat)

    geometries = []
    for i, points in enumerate(meshes):
        displacement = [[gap * (i + 1), 0, 0]] if (save_dir is None) else 0
        points = np.dot(points, rot_mat.T)
        if (faces is not None and faces[i] is not None):
            geometry = o3d.geometry.TriangleMesh()
            geometry.vertices = o3d.utility.Vector3dVector(points + displacement)
            geometry.triangles = o3d.utility.Vector3iVector(deepcopy(faces[i]))
            geometry.compute_vertex_normals()
        else:
            geometry = o3d.geometry.PointCloud()
            geometry.points = o3d.utility.Vector3dVector(points + displacement)
            if colors is not None and colors[i] is not None:
                geometry.colors = o3d.utility.Vector3dVector(colors[i])
            else:
                geometry.colors = o3d.utility.Vector3dVector(np.ones(points.shape) * 0.7)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            name = names[i]
            if angle is not None:
                name += f'_{angle}'
            zoom = zooms[i] if zooms is not None else None
            save_vis(geometry, name, save_dir, dataset_mode, zoom)
        else:
            geometries.append(geometry)
    if save_dir is None:
        o3d.visualization.draw_geometries(geometries)
    else:
        for group in ['points', 'mesh']:
            imgs = []
            suffix = '.jpg' if group == 'points' else '_mesh.jpg'
            for n in names:
                img_p = os.path.join(save_dir, n + suffix)
                if os.path.exists(img_p):
                    if paper_order:
                        imgs.append(cv2.imread(img_p))
                    else:
                        imgs.append(Image.open(img_p))

            if len(imgs) > 0:
                fpath = os.path.join(save_dir, os.path.basename(save_dir) + suffix)
                if paper_order:
                    top = 100
                    bottom = 900
                    out_img = np.concatenate([img[top:bottom] for img in imgs], axis=1)
                    cv2.imwrite(fpath, out_img)
                else:
                    out_w = sum([img.size[0] for img in imgs])
                    out_h = max([img.size[1] for img in imgs])
                    out_img = Image.new('RGB', (out_w, out_h))
                    out_img.paste(imgs[0], (0, 0))
                    start = imgs[0].size[0]
                    for i, img in enumerate(imgs[1:]):
                        out_img.paste(img, (start, 0))
                        start += img.size[0]
                    out_img.save(fpath)

def save_vis(mesh, path, save_dir, dataset_mode, zoom):
    import open3d as o3d

    if dataset_mode == 'human':
        w, h = 700, 1000
    elif dataset_mode == 'animal':
        w, h = 1000, 1000

    if 'Mesh' in type(mesh).__name__:
        path += '_mesh'
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h)
    vis.add_geometry(mesh)
    if zoom is not None:
        ctr = vis.get_view_control()
        ctr.set_zoom(zoom)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(os.path.join(save_dir, path + '.jpg'))
    vis.destroy_window()