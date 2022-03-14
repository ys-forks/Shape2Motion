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
    mat_i = 6
    loadpath1 = f'../main/train_data_stage_2/train_stage_2_data_{mat_i}.mat'
    train_data = sio.loadmat(loadpath1)['Training_data']

    loadpath1 = f'../main/test_result_s_2/test_s_2_pred__data_{mat_i}.mat'
    test_data_s2 = sio.loadmat(loadpath1)
    pred_dof_score = test_data_s2['pred_dof_score_val']

    loadpath = f'../main/test_result_s_3/test_s_3_pred__data_{mat_i}.mat'
    test_data = sio.loadmat(loadpath)
    pred_proposal = test_data['pred_proposal']
    pred_dof_regression = test_data['pred_dof_regression']
    print(np.amax(pred_proposal))
    print(np.amin(pred_proposal))

    proposal = tf.argmax(pred_proposal,axis=2,output_type = tf.int32)
    proposal = tf.greater(tf.cast(proposal,tf.float32), 0.5)

    print(len(proposal))
    inst_id = 0
    tmp_xyz = None
    predictions = {
        'mask': [],
        'score': [],
        'motion': [],
        'motion_type': [],
    }
    for idx in range(0, 11, 1):
        print(idx)
        instance_data = train_data[idx][0]
        
        inputs_all = instance_data['inputs_all'][0, 0]

        pred_score = pred_dof_score[idx]
        np.unique(pred_score, return_counts=True)

        xyz = tf.slice(inputs_all, [0,0], [-1,3]).numpy()
        normals = tf.slice(inputs_all, [0,3], [-1,3]).numpy()

        if tmp_xyz is None:
            tmp_xyz = xyz

        if not np.all(tmp_xyz == xyz):
            tmp_xyz = xyz
            print('split', idx)
            inst_id += 1
            break

        pred_motion_base = instance_data['dof_pred'][0, 0]
        pred_dof_mask = instance_data['dof_mask'][0, 0].flatten()
        gt_score = instance_data['dof_score'][0, 0].flatten()

        score = np.sum(gt_score[pred_dof_mask.astype(bool)])
        print(gt_score)
        predictions['score'].append(score)

        pred_mask = proposal[idx].numpy()
        predictions['mask'].append(pred_mask)
        pred_motion_res = pred_dof_regression[idx]

        selected_motions = pred_motion_base[pred_dof_mask.astype(bool)]
        selected_motions[:, :6] += pred_motion_res

        # gt viz
        if True:
            viz = Viewer(vertices=xyz, mask=pred_mask.astype(int))
            arrow_poss = selected_motions[:, :3]
            arrow_dirs = selected_motions[:, 3:6]
            print(selected_motions[:, 6])
            arrow_dirs = arrow_dirs / np.linalg.norm(arrow_dirs, axis=1).reshape(-1, 1)
            cm = plt.get_cmap('jet')
            color = cm(pred_score[pred_dof_mask.astype(bool)])
            print(color)

            viz.add_trimesh_arrows(arrow_poss[::3], arrow_dirs[::3], colors=color[::3], radius=0.005, length=0.1)
            viz.show()
    cluster = {}
    parts = 0
    overlap_thresh = 0.5
    masks = predictions['mask']
    pdb.set_trace()
    for i, mask in enumerate(masks):
        if i == 0:
            cluster[parts] = [i]
            parts += 1
            continue
        merge = False
        for k, v in cluster.items():
            inter = np.logical_and(mask, masks[v[0]])
            union = np.logical_or(mask, masks[v[0]])
            overlap = np.sum(inter) / np.sum(union)
            print(overlap)
            if overlap > overlap_thresh:
                merge = True
                cluster[k].append(i)
                break
        if not merge:
            cluster[parts] = [i]
            parts += 1
    print(list(cluster.keys()))

    max_score = -1
    best_ones = {}
    for k, v in cluster.items():
        for i in v:
            score = predictions['score'][i]
            if score > max_score:
                max_score = score
                best_ones[k] = i
    print(best_ones)
    print(predictions['score'])




