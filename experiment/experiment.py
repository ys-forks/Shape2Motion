import mat73
import scipy.io as sio
import tensorflow as tf
import numpy as np
import open3d as o3d
import pdb
from viewer import Viewer
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import trimesh
# tf.enable_eager_execution()

COLOR20 = np.array(
    [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
     [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
     [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
     [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])

if __name__ == "__main__":
    loadpath = '../result/train_data/training_data_1.mat'
    train_data = sio.loadmat(loadpath)['Training_data']
    print(len(train_data))
    instance_data = train_data[6][0]
    # print(instance_data.keys())
    inputs_all = instance_data['inputs_all'][0,0]
    xyz = tf.slice(inputs_all, [0,0], [-1,3]).numpy()
    normals = tf.slice(inputs_all, [0,3], [-1,3]).numpy()

    anchor = instance_data['core_position'][0,0].reshape(-1)
    motion_direction_class = instance_data['motion_direction_class'][0,0].reshape(-1)
    motion_direction_delta = instance_data['motion_direction_delta'][0,0]
    anchor_displ_param = instance_data['motion_position_param'][0,0]
    motion_type = instance_data['motion_dof_type'][0,0]
    all_direction_kmeans = instance_data['all_direction_kmeans'][0,0]
    dof_matrix = instance_data['dof_matrix'][0,0]
    proposal = instance_data['proposal'][0,0]
    similar_matrix = instance_data['similar_matrix'][0,0]

    print(normals.shape)
    print(anchor.shape)
    print(motion_direction_class.shape)
    print(np.unique(motion_direction_class, return_counts=True))
    print(motion_direction_delta.shape)
    print(anchor_displ_param.shape)
    print(motion_type.shape)

    print(np.unique(anchor, return_counts=True, axis=0))
    # pdb.set_trace()
    pid = 2
    # viz = Viewer(vertices=xyz, mask=proposal[pid+1].astype(int))
    # arrow_pos = dof_matrix[pid][:3]
    # r = R.from_quat(dof_matrix[pid][3:])
    # arrow_dir = r.as_rotvec()
    # arrow_dir = arrow_dir / np.linalg.norm(arrow_dir)
    # viz.add_trimesh_arrows([arrow_pos], [arrow_dir])
    # viz.show()

    mask = np.zeros_like(proposal)
    for i in range(len(proposal)):
        mask[i] = proposal[i]*i
    mask = np.sum(mask, axis=0)

    # gt viz
    viz = Viewer(vertices=xyz, mask=mask.astype(int))
    arrow_poss = dof_matrix[:, :3]
    arrow_dirs = dof_matrix[:, 3:6]
    arrow_dirs = arrow_dirs / np.linalg.norm(arrow_dirs, axis=1).reshape(-1, 1)
    arrow_colors = []
    for i in range(len(arrow_dirs)):
        a_type = dof_matrix[i, -1]
        arrow_colors.append(Viewer.rgba_by_index(a_type.astype(int)))
    arrow_colors = np.asarray(arrow_colors)
    # pdb.set_trace()

    anchor_pts = xyz[anchor.astype(bool)] + anchor_displ_param[anchor.astype(bool)]
    for i in range(len(anchor_pts)):
        t = np.eye(4)
        t[:3, 3] = anchor_pts[i]
        sphere = trimesh.creation.icosphere(radius=0.005, color=[0.0, 1.0, 0.0])
        sphere.apply_transform(t)
        viz.add_trimesh(sphere)

    viz.add_trimesh_arrows(arrow_poss, arrow_dirs, colors=arrow_colors, radius=0.005, length=0.1)
    viz.show()