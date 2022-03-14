import mat73
import scipy.io as sio
import tensorflow as tf
import numpy as np
import open3d as o3d
from viewer import Viewer
import pdb
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# tf.enable_eager_execution()

COLOR20 = np.array(
    [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
     [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
     [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
     [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])

if __name__ == "__main__":
    mat_i = 6
    loadpath1 = f'../main/train_data_stage_2/train_stage_2_data_{mat_i}.mat'
    train_data = sio.loadmat(loadpath1)['Training_data']

    loadpath = f'../main/test_result_s_3/test_s_3_pred__data_{mat_i}.mat'
    test_data = sio.loadmat(loadpath)
    pred_proposal = test_data['pred_proposal']
    pred_dof_regression = test_data['pred_dof_regression']

    proposal = tf.argmax(pred_proposal,axis=2,output_type = tf.int32)
    proposal = tf.greater(tf.cast(proposal,tf.float32),0.5)

    print(len(proposal))
    for idx in range(0, len(proposal), 1):
        print(idx)
        instance_data = train_data[idx][0]
        
        inputs_all = instance_data['inputs_all'][0, 0]

        xyz = tf.slice(inputs_all, [0,0], [-1,3]).numpy()
        normals = tf.slice(inputs_all, [0,3], [-1,3]).numpy()

        pred_motion_base = instance_data['dof_pred'][0, 0]
        pred_dof_mask = instance_data['dof_mask'][0, 0].flatten()
        pred_score = instance_data['dof_score'][0, 0].flatten()

        pred_mask = proposal[idx].numpy()
        pred_motion_res = pred_dof_regression[idx]

        selected_motions = pred_motion_base[pred_dof_mask.astype(bool)]
        selected_motions[:, :6] += pred_motion_res

        print(np.sum(pred_score[pred_dof_mask.astype(bool)]))

        # gt viz
        viz = Viewer(vertices=xyz, mask=pred_mask.astype(int))
        arrow_poss = selected_motions[:, :3]
        arrow_dirs = selected_motions[:, 3:6]
        arrow_dirs = arrow_dirs / np.linalg.norm(arrow_dirs, axis=1).reshape(-1, 1)
        # pdb.set_trace()
        cm = plt.get_cmap('jet')
        color = cm(pred_score[pred_dof_mask.astype(bool)])

        viz.add_trimesh_arrows(arrow_poss[::3], arrow_dirs[::3], colors=color[::3], radius=0.005, length=0.1)
        viz.show()

        # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        # pcd.normals = o3d.utility.Vector3dVector(normals)
        # colors = np.asarray([[0.3, 0.3, 0.3]] * len(xyz))
        # for i in range(len(motion_direction_class)):
        #     if anchor[i]:
        #         colors[i] = COLOR20[int(motion_direction_class[i])] / 255.0
        #     pcd.colors = o3d.utility.Vector3dVector(colors)

        # o3d.io.write_point_cloud('test.ply', pcd)
        
        # colors = np.asarray([[0.3, 0.3, 0.3]] * len(xyz))
        # for i in range(len(proposal)):
        #     colors[proposal[i]] = COLOR20[i] / 255.0
        #     pcd.colors = o3d.utility.Vector3dVector(colors)

        # o3d.io.write_point_cloud('test1.ply', pcd)