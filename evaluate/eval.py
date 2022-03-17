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
            'gt_motion': [],
            'gt_mask': [],
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

            gt_proposal = instance_data['GT_proposal_nx'][0, 0].flatten()
            gt_dof = instance_data['GT_dof'][0, 0].flatten()
            predictions['gt_motion'].append(gt_dof)
            predictions['gt_mask'].append(gt_proposal)

            pred_motion_base = instance_data['dof_pred'][0, 0]
            pred_dof_mask = instance_data['dof_mask'][0, 0].flatten()
            gt_score = instance_data['dof_score'][0, 0].flatten()

            selected_scores = pred_score[pred_dof_mask.astype(bool)]
            predictions['score'].append(selected_scores)
            predictions['xyz'].append(xyz)

            pred_mask = proposal[idx].numpy()
            predictions['mask'].append(pred_mask)
            pred_motion_res = pred_dof_regression[idx]

            selected_motions = pred_motion_base[pred_dof_mask.astype(bool)]
            selected_motions[:, :6] += pred_motion_res
            arrow_dirs = selected_motions[:, 3:6]
            arrow_dirs = arrow_dirs / np.linalg.norm(arrow_dirs, axis=1).reshape(-1, 1)
            selected_motions[:, 3:6] = arrow_dirs
            predictions['motion'].append(selected_motions)

            assert len(selected_motions) == len(selected_scores)

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
        score_thresh = 0.7
        for k, v in cluster.items():
            best_motion[k] = {'translation': [], 'rotation': []}
            for i in v:
                motions = predictions['motion'][i]
                for m_i, m in enumerate(motions):
                    if predictions['score'][i][m_i] < score_thresh:
                        continue
                    if m[6] == 2:
                        if len(best_motion[k]['translation']) > 0:
                            tmp_mi = best_motion[k]['translation'][0]
                            tmp_score = predictions['score'][tmp_mi[0]][tmp_mi[1]]
                            if predictions['score'][i][m_i] > tmp_score:
                                best_motion[k]['translation'][0] = (i, m_i)
                        else:
                            best_motion[k]['translation'].append((i, m_i))
                    
                    if m[6] == 1:
                        if len(best_motion[k]['rotation']) > 0:
                            merge = False
                            for tmp_i, tmp_mi in enumerate(best_motion[k]['rotation']):
                                angle = np.dot(m[3:6], predictions['motion'][tmp_mi[0]][tmp_mi[1]][3:6])
                                # pdb.set_trace()
                                if angle > np.cos(np.pi / 4.0):
                                    merge = True
                                    tmp_score = predictions['score'][tmp_mi[0]][tmp_mi[1]]
                                    if predictions['score'][i][m_i] > tmp_score:
                                        best_motion[k]['rotation'][tmp_i] = (i, m_i)
                            if merge == False:
                                best_motion[k]['rotation'].append((i, m_i))
                        else:
                            best_motion[k]['rotation'].append((i, m_i))

        print(best_motion)

        # eval
        ious = []
        for k, v in best_segm.items():
            gt_mask = predictions['gt_mask'][v]
            inter = np.logical_and(gt_mask, masks[v])
            union = np.logical_or(gt_mask, masks[v])
            iou = np.sum(inter) / np.sum(union)
            ious.append(iou)
        mean_iou = np.mean(ious)
        print('mean part iou', round(mean_iou, 4))

        orig_errs = []
        dir_errs = []
        type_accs = []
        for k, v in best_motion.items():
            scores = []
            max_t_score = -1
            for t in v['translation']:
                scores.append(predictions['score'][t[0]][t[1]])
            if len(scores) > 0:
                max_t_score_i = np.argmax(scores)
                max_t_score = scores[max_t_score_i]

            scores = []
            max_r_score = -1
            for r in v['rotation']:
                scores.append(predictions['score'][r[0]][r[1]])
            if len(scores) > 0:
                max_r_score_i = np.argmax(scores)
                max_r_score = scores[max_r_score_i]

            if max_t_score < 0 and max_r_score < 0:
                print('no motion predicted')
                continue
            
            m_type = 1 if max_r_score_i >= max_t_score else 2
            if m_type == 1:
                pred_m_i = v['rotation'][max_r_score_i]
            else:
                pred_m_i = v['translation'][max_r_score_i]
            pred_m = predictions['motion'][pred_m_i[0]][pred_m_i[1]]
            gt_motion = predictions['gt_motion'][pred_m_i[0]]

            p_orig = pred_m[:3]
            g_orig = gt_motion[:3]

            dis = np.linalg.norm(np.cross(pred_m[3:6], p_orig-g_orig)) / np.linalg.norm(pred_m[3:6])
            orig_errs.append(dis)

            p_dir = pred_m[3:6]
            g_dir = gt_motion[3:6]

            angle = np.arccos(np.clip(np.cos(p_dir, g_dir), -1, 1))
            dir_errs.append(angle)

            gt_type = gt_motion[6]
            type_accs.append(m_type == gt_type)
        
        mean_orig_err = np.mean(orig_errs)
        print('mean joint origin err', round(mean_orig_err, 4))
        mean_dir_err = np.mean(dir_errs)
        print('mean joint direction err (degree)', round(mean_dir_err,4))
        type_acc = np.sum(type_accs) / len(best_motion)
        print('mean joint type accuracy', round(type_acc, 4))
        
        object_masks = []
        for k, v in best_segm.items():
            object_masks.append(masks[v].astype(int) * (k+1))
        object_mask = np.sum(object_masks, axis=0)

        RED = [1.0,0,0,1.0]
        BLUE = [0,0,1.0,1.0]

        viz = Viewer(vertices=last_xyz, mask=object_mask.astype(int))
        arrow_poss = []
        arrow_dirs = []
        arrow_colors = []
        for k, v in best_motion.items():
            for t_mi in v['translation']:
                m = predictions['motion'][t_mi[0]][t_mi[1]]
                arrow_pos = m[:3]
                arrow_dir = m[3:6]
                arrow_poss.append(arrow_pos)
                arrow_dirs.append(arrow_dir)
                arrow_colors.append(BLUE)
            for t_mi in v['rotation']:
                m = predictions['motion'][t_mi[0]][t_mi[1]]
                arrow_pos = m[:3]
                arrow_dir = m[3:6]
                arrow_poss.append(arrow_pos)
                arrow_dirs.append(arrow_dir)
                arrow_colors.append(RED)

        viz.add_trimesh_arrows(arrow_poss, arrow_dirs, colors=arrow_colors, radius=0.005, length=0.2)
        viz.show()

        if idx == len(proposal)-1 :
            break
        break






