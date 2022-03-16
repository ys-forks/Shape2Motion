import mat73
import scipy.io as sio
import tensorflow as tf
import numpy as np
import open3d as o3d
from viewer import Viewer
import pdb
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

DEBUG = False

if __name__ == "__main__":
    mat_i = 1
    loadpath1 = f'../main/train_data_stage_2/train_stage_2_data_{mat_i}.mat'
    train_data = sio.loadmat(loadpath1)['Training_data']

    loadpath1 = f'../main/test_result_s_2/test_s_2_pred__data_{mat_i}.mat'
    test_data_s2 = sio.loadmat(loadpath1)
    pred_dof_score = test_data_s2['pred_dof_score_val']

    loadpath = f'../main/test_result_s_3/test_s_3_pred__data_{mat_i}.mat'
    test_data = sio.loadmat(loadpath)
    pred_proposal = test_data['pred_proposal']
    pred_dof_regression = test_data['pred_dof_regression']

    proposal = tf.argmax(pred_proposal,axis=2,output_type = tf.int32)
    proposal = tf.greater(tf.cast(proposal,tf.float32), 0.5)

    t_idx = 0
    while (t_idx < len(proposal)):
        print(t_idx)
        last_xyz = None
        predictions = {
            'mask': [],
            'xyz': [],
            'score': [],
            'motion': [],
            'motion_type': [],
        }
        for idx in range(t_idx, len(proposal), 1):
            print(idx)
            instance_data = train_data[idx][0]
            
            inputs_all = instance_data['inputs_all'][0, 0]

            pred_score = pred_dof_score[idx].flatten()

            xyz = tf.slice(inputs_all, [0,0], [-1,3]).numpy()
            normals = tf.slice(inputs_all, [0,3], [-1,3]).numpy()

            if last_xyz is None:
                last_xyz = xyz
            
            if not np.array_equal(last_xyz, xyz):
                t_idx = idx
                break

            last_xyz = xyz

            pred_motion_base = instance_data['dof_pred'][0, 0]
            pred_dof_mask = instance_data['dof_mask'][0, 0].flatten()
            gt_score = instance_data['dof_score'][0, 0].flatten()

            predictions['score'].append(pred_score)
            predictions['xyz'].append(xyz)

            pred_mask = proposal[idx].numpy()
            predictions['mask'].append(pred_mask)
            pred_motion_res = pred_dof_regression[idx]

            selected_motions = pred_motion_base[pred_dof_mask.astype(bool)]
            selected_motions[:, :6] += pred_motion_res
            predictions['motion'].append(selected_motions)

            if DEBUG:
                viz = Viewer(vertices=xyz, mask=pred_mask.astype(int))
                arrow_poss = selected_motions[:, :3]
                arrow_dirs = selected_motions[:, 3:6]
                
                arrow_dirs = arrow_dirs / np.linalg.norm(arrow_dirs, axis=1).reshape(-1, 1)
                cm = plt.get_cmap('jet')
                color = cm(pred_score[pred_dof_mask.astype(bool)])
                # pdb.set_trace()

                viz.add_trimesh_arrows(arrow_poss[::3], arrow_dirs[::3], colors=color[::3], radius=0.005, length=0.1)
                viz.show()
        
        cluster = {}
        parts = 0
        overlap_thresh = 0.5
        masks = predictions['mask']
        
        for i, mask in enumerate(masks):
            # viz = Viewer(vertices=predictions['xyz'][i], mask=mask.astype(int))
            # viz.show()
            if i == 0:
                cluster[parts] = [i]
                parts += 1
                continue
            merge = False
            for k, v in cluster.items():
                inter = np.logical_and(mask, masks[v[0]])
                union = np.logical_or(mask, masks[v[0]])
                overlap = np.sum(inter) / np.sum(union)
                if overlap > overlap_thresh:
                    merge = True
                    cluster[k].append(i)
                    break
            if not merge:
                # pdb.set_trace()
                cluster[parts] = [i]
                parts += 1
        print(list(cluster.keys()))

        # if True:
        #     for k, v in cluster.items():
        #         viz = Viewer(vertices=last_xyz, mask=masks[v[0]].astype(int))
        #         viz.show()

        best_segm = {}
        for k, v in cluster.items():
            max_score = -1
            for i in v:
                score = np.amax(predictions['score'][i])
                if score > max_score:
                    max_score = score
                    best_segm[k] = i
        print(best_segm)

        best_motion = {}
        for k, v in cluster.items():
            max_score = -1
            best_motion = predictions['motion'][best_segm[k]]
            best_motion[k].append({best_segm[k]: best_motion})
            for i in v:
                motion = predictions['motion'][i]
                motion_type = motion[6]
                # rotatition
                if motion_type == 1:
                    motion_dir = motion[3:6]
                    for m in best_motion[k].values():
                        tmp_motion = list(m.values())[0]
                        if motion_type != tmp_motion[6]:
                            continue
                        if np.cos(motion[3:6], tmp_motion[3:6])
                    best_motion[k].append({i: best_motion})


        if idx == len(proposal)-1 :
            break
        break






