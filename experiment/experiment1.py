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
    loadpath = '../main/train_data_stage_2/train_stage_2_data_1.mat'
    train_data = sio.loadmat(loadpath)['Training_data']
    print(len(train_data))
    
    for idx in range(0, len(train_data), 5):
        print(idx)
        instance_data = train_data[idx][0]
        # print(instance_data.keys())
        
        inputs_all = instance_data['inputs_all'][0, 0]
        xyz = tf.slice(inputs_all, [0,0], [-1,3]).numpy()
        normals = tf.slice(inputs_all, [0,3], [-1,3]).numpy()

        # ['GT_dof', 'GT_proposal_nx', 'dof_mask', 'dof_pred', 'dof_score', 'inputs_all', 'proposal_nx']

        gt_proposal = instance_data['GT_proposal_nx'][0, 0].flatten()
        gt_dof = instance_data['GT_dof'][0, 0].flatten()

        pred_anchor = instance_data['dof_mask'][0, 0].flatten()
        pred_dof = instance_data['dof_pred'][0, 0]
        pred_score = instance_data['dof_score'][0, 0].flatten()
        pred_proposal = instance_data['proposal_nx'][0, 0].flatten()

        # gt viz
        viz = Viewer(vertices=xyz, mask=gt_proposal.astype(int))
        arrow_pos = gt_dof[:3]
        arrow_dir = gt_dof[3:6]
        print(gt_dof[6])
        # pdb.set_trace()
        arrow_dir = arrow_dir / np.linalg.norm(arrow_dir)
        viz.add_trimesh_arrows([arrow_pos], [arrow_dir], length=1.0)
        viz.show()

        viz.reset()
        viz.add_geometry(vertices=xyz, mask=pred_proposal.astype(int))
        motions = pred_dof[pred_anchor > 0]
        arrow_poss = motions[:, :3]
        arrow_dirs = motions[:, 3:6]
        arrow_dirs = arrow_dirs / np.linalg.norm(arrow_dirs, axis=1).reshape(-1,1)
        # pdb.set_trace()
        cm = plt.get_cmap('jet')
        color = cm(pred_score[pred_anchor > 0])
        print(motions[:, 6])
        print(np.unique(pred_score[pred_anchor > 0]))
        # color *= 255
        # color = color.astype('uint8')
        viz.add_trimesh_arrows(arrow_poss[::5], arrow_dirs[::5], colors=color[::5], radius=0.005, length=0.1)
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