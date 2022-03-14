import mat73
import scipy.io as sio
import tensorflow as tf
import numpy as np
import open3d as o3d
from viewer import Viewer
import pdb
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    mat_idx = 1
    loadpath = f'../main/test_result_s_2/test_s_2_pred__data_{mat_idx}.mat'

    test_data = sio.loadmat(loadpath)
    pred_dof_score_val = test_data['pred_dof_score_val']
    all_feat = test_data['all_feat']

    loadpath = f'../main/train_data_stage_2/train_stage_2_data_{mat_idx}.mat'
    train_data = sio.loadmat(loadpath)['Training_data']
    print(len(train_data))
    assert len(train_data) == len(pred_dof_score_val)
    
    for idx in range(0, len(train_data), 2):
        print(idx)
        instance_data = train_data[idx][0]
        # print(instance_data.keys())
        
        inputs_all = instance_data['inputs_all'][0, 0]
        xyz = tf.slice(inputs_all, [0,0], [-1,3]).numpy()
        normals = tf.slice(inputs_all, [0,3], [-1,3]).numpy()
        # ['GT_dof', 'GT_proposal_nx', 'dof_mask', 'dof_pred', 'dof_score', 'inputs_all', 'proposal_nx']

        gt_proposal = instance_data['GT_proposal_nx'][0, 0].flatten()
        print(gt_proposal)
        gt_dof = instance_data['GT_dof'][0, 0].flatten()

        pred_anchor = instance_data['dof_mask'][0, 0].flatten()
        pred_dof = instance_data['dof_pred'][0, 0]
        gt_score = instance_data['dof_score'][0, 0].flatten()
        pred_proposal = instance_data['proposal_nx'][0, 0].flatten()

        pred_score = pred_dof_score_val[idx].flatten()

        # gt viz
        viz = Viewer(vertices=xyz, mask=gt_proposal.astype(int))
        arrow_pos = gt_dof[:3]
        arrow_dir = gt_dof[3:6]
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
        color = cm(gt_score[pred_anchor > 0])
        # color *= 255
        # color = color.astype('uint8')
        # pdb.set_trace()
        viz.add_trimesh_arrows(arrow_poss[::3], arrow_dirs[::3], colors=color[::3], radius=0.005, length=0.1)
        viz.show()

        viz.reset()
        viz.add_geometry(vertices=xyz, mask=pred_proposal.astype(int))
        cm = plt.get_cmap('jet')
        color = cm(pred_score[pred_anchor > 0])
        viz.add_trimesh_arrows(arrow_poss[::3], arrow_dirs[::3], colors=color[::3], radius=0.005, length=0.1)
        viz.show()