import enum
import os
from pathlib import Path
import glob
import torch
import pymeshlab
import trimesh
import numpy as np
import pandas as pd
from dgl.geometry import farthest_point_sampler
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import pdb
import json
import treet
import scipy.io as sio
import hdf5storage
import copy
from viewer import Viewer

DEBUG = False

all_direction_kmeans = np.array([
    [ 1.        ,  0.        ,  0.        ],
    [-1.        ,  0.        ,  0.        ],
    [ 0.        ,  1.        ,  0.        ],
    [ 0.        , -1.        ,  0.        ],
    [ 0.        ,  0.        ,  1.        ],
    [ 0.        ,  0.        , -1.        ],
    [ 0.57735026,  0.57735026,  0.57735026],
    [ 0.57735026,  0.57735026, -0.57735026],
    [ 0.57735026, -0.57735026,  0.57735026],
    [ 0.57735026, -0.57735026, -0.57735026],
    [-0.57735026,  0.57735026,  0.57735026],
    [-0.57735026,  0.57735026, -0.57735026],
    [-0.57735026, -0.57735026,  0.57735026],
    [-0.57735026, -0.57735026, -0.57735026]
])

dataset_path = '/localhome/yma50/Development/Shape2Motion/dataset/shape2motion'
result_path = '/localhome/yma50/Development/Shape2Motion/result'
obj_folder = 'part_objs'
motion_file = 'motion_attributes.json'

num_points = 4096

np.random.seed(0)

from urdfpy import URDF, JointLimit

# override attributes to make effort, velocity optional
JointLimit._ATTRIBS = {
    'effort': (float, False),
    'velocity': (float, False),
    'lower': (float, False),
    'upper': (float, False),
}
# set default values
JointLimit.effort = 1.0
JointLimit.velocity = 1000

def get_sub_folders(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def split_data():
    cat_folders = get_sub_folders(dataset_path)

    obj_df = []
    for cat in cat_folders:
        inst_folders = get_sub_folders(os.path.join(dataset_path, cat))
        for inst in inst_folders:
            df_row = pd.DataFrame([[cat, inst]], columns=['category', 'instanceId'])
            obj_df.append(df_row)

    obj_df = pd.concat(obj_df)

    split_df = obj_df['category'].drop_duplicates().to_frame()
    train_percent = 0.8
    seed = 0
    set_size = len(split_df)
    train_set, test_set = np.split(
        split_df.sample(frac=1.0, random_state=seed),
        [int(train_percent * set_size)]
    )
    train = train_set.merge(obj_df, how='left', on='category')
    test = test_set.merge(obj_df, how='left', on='category')
    split_info = pd.concat([train, test], keys=['train', 'test'], names=['set', 'index'])
    split_info.to_csv('/localhome/yma50/Development/Shape2Motion/dataset/split.csv')
    return split_info


def load_urdf(urdf_path):
    urdf_data = URDF.load(urdf_path)
    assert urdf_data, "URDF data is empty!"

    link_infos = []
    link_meshes = []
    link_idx = 0

    joint_infos = []
    augm = {}
    for joint in urdf_data.joints:
        joint_info = {
            'name': joint.name,
            'type': joint.joint_type,
            'parent': joint.parent,
            'child': joint.child,
            'axis': joint.axis.tolist(),
            'pose2link': joint.origin.flatten(order='F').tolist()
        }
        augm[joint.name] = np.concatenate([0], np.random.random_sample(9)) * np.pi / 2
    joint_infos.append(joint_info)

    for i in range(10):
        augm_cfg = {}
        for k, v in augm.items():
            augm_cfg[k] = v[i]

        urdf_data.show(cfg=augm_cfg)
        for link, link_abs_pose in urdf_data.link_fk(cfg=augm_cfg).items():
            link_info = {'name': link.name}
            fk_visual = urdf_data.visual_trimesh_fk(links=[link])
            is_virtual = not bool(fk_visual)
            link_info['virtual'] = is_virtual
            link_info['abs_pose'] = link_abs_pose.flatten(order='F').tolist()
            if not is_virtual:
                link_mesh = trimesh.base.Trimesh()
                for mesh, mesh_abs_pose in fk_visual.items():
                    mesh.apply_transform(mesh_abs_pose)
                    # remove texture visual
                    mesh.visual = trimesh.visual.create_visual()
                    link_mesh += mesh
                # part mesh visualization
                color = Viewer.rgba_by_index(link_idx)
                color[-1] = 0.8
                link_mesh.visual.vertex_colors = color
                if DEBUG:
                    link_mesh.show()
                link_info['part_index'] = link_idx
                link_meshes.append(link_mesh)
                link_idx += 1
            else:
                link_info['part_index'] = -1
            link_infos.append(link_info)

def generate_gt(urdf_path):
    # points, normals, segms, part_info = load_mesh(mesh_path)
    # nodes = parse_motions(motion_path)
    print(urdf_path)
    load_urdf(str(urdf_path))

    # assert len(part_info) == len(nodes)+1

    # instance_data = {
    #     'inputs_all': np.column_stack((points, normals)),
    #     'core_position': np.zeros(num_points),
    #     'motion_direction_class': np.zeros(num_points),
    #     'motion_direction_delta': np.zeros((num_points, 3)),
    #     'motion_position_param': np.zeros((num_points, 3)),
    #     'motion_dof_type': np.zeros(num_points),
    #     'all_direction_kmeans': all_direction_kmeans,
    #     'dof_matrix': np.zeros((len(nodes), 7)),
    #     'proposal': np.zeros((len(part_info), num_points)),
    #     'similar_matrix': np.zeros((num_points, num_points))
    # }
    # for i, node in enumerate(nodes):
    #     motion_type = 1 if node['motion_type'] == 'rotation' else 2
    #     center = np.asarray(node['center'])
    #     direction = np.asarray(node['direction'])
    #     direction = direction / np.linalg.norm(direction)

    #     center_disp = points - center
    #     distance = np.linalg.norm(np.cross(center_disp, direction), axis=1)
    #     anchor_indices = np.argsort(distance)[:30]
    #     instance_data['core_position'][anchor_indices] = 1

    #     rad = np.arccos(np.clip(np.dot(all_direction_kmeans, direction), -1.0, 1.0))
    #     direction_class = np.argmin(rad)
    #     direction_delta = direction - all_direction_kmeans[direction_class]
    #     position_param = center_disp.dot(direction).reshape(-1, 1) * direction - center_disp
    #     instance_data['motion_direction_class'][anchor_indices] = direction_class + 1
    #     instance_data['motion_direction_delta'][anchor_indices] = direction_delta
    #     instance_data['motion_position_param'][anchor_indices] = position_param[anchor_indices]
    #     instance_data['motion_dof_type'][anchor_indices][instance_data['motion_dof_type'][anchor_indices] != motion_type] += motion_type
    #     instance_data['dof_matrix'][i] = np.concatenate((center, direction, [motion_type]))
    #     segm_id = part_info[node['dof_name']]
    #     instance_data['proposal'][i+1] = segms == segm_id
    # # pdb.set_trace()
    # instance_data['proposal'][0] = segms == part_info['none_motion']

    # for i in range(num_points):
    #     instance_data['similar_matrix'][i] = segms == segms[i]

    # instance_data['core_position'] = instance_data['core_position'].astype(bool)
    # instance_data['proposal'] = instance_data['proposal'].astype(bool)
    # instance_data['similar_matrix'] = instance_data['similar_matrix'].astype(bool)
    
    # return instance_data

def export_data(batch, output):
    batch_gt = []
    for i, row in batch.iterrows():
        cat, inst = row['category'], row['instanceId']
        print(f'{cat}/{inst}')
        mesh_path = Path(dataset_path) / cat / inst / 'part_objs'
        motion_path = Path(dataset_path) / cat / inst / 'motion_attributes.json'
        urdf_path = Path(result_path) / 'urdf' / cat / inst / 'syn.urdf'
        instance_data = generate_gt(urdf_path)
        batch_gt.append([instance_data])
    output_mat = {'Training_data': batch_gt}
    # with open(str(output), 'w+') as f:
    #     hdf5storage.savemat(str(output), output_mat, format='7.3')
    # sio.savemat(str(output), output_mat)
    print(f'export to {output}')

def generate_data(df, set='train', categories=['car']):
    data_df = df.loc[set]
    data_df = data_df[data_df['category'].isin(categories)]
    print(len(data_df))
    B = 16
    if set == 'test':
        B = 1
    count = 0
    (Path(result_path) / f'{set}_data').mkdir(parents=True, exist_ok=True)
    for i in range(0, len(data_df), B):
        batch = data_df[i:i+B]
        output_path = Path(result_path) / f'{set}_data' / f'{set}ing_data_{count+1}.mat'
        export_data(batch, output_path)
        count += 1
        

if __name__ == '__main__':
    data_df = split_data()

    generate_data(data_df, 'train')
    # generate_data(data_df, 'test')