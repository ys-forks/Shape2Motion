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
from multiscan.utils import io

DEBUG = False
COLOR20 = np.array(
    [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
     [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
     [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
     [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])

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


def load_mesh(mesh_path):
    obj_files = glob.glob(str(mesh_path / '*.obj'))

    part_info = {}
    vertices = None
    faces = None
    vertex_normals = None
    face_normals = None
    colors = None
    segms = None

    meshes = []

    for i, obj_f in enumerate(obj_files):
        s = trimesh.load(obj_f, force='scene')
        part_name = os.path.splitext(os.path.basename(obj_f))[0]
        part_info[part_name] = i
        if len(s.geometry) == 0:
            part_info[part_name] = -1
            continue

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(obj_f)

        cur_mesh = ms.current_mesh()
        np_vertices = cur_mesh.vertex_matrix()
        np_faces = cur_mesh.face_matrix()
        np_vertex_normals = cur_mesh.vertex_normal_matrix()
        np_face_normals = cur_mesh.face_normal_matrix()
        np_colors = cur_mesh.vertex_color_matrix()

        # vertices = np_vertices if vertices is None else np.vstack((vertices, np_vertices))
        # faces = np_faces if faces is None else np.vstack((faces, np_faces))
        # vertex_normals = np_vertex_normals if vertex_normals is None else np.vstack((vertex_normals, np_vertex_normals))
        # face_normals = np_face_normals if face_normals is None else np.vstack((face_normals, np_face_normals))
        # colors = np_colors if colors is None else np.vstack((colors, np_colors))
        m = trimesh.Trimesh(vertices=np_vertices, faces=np_faces, face_normals=np_face_normals, vertex_normals=np_vertex_normals, vertex_colors=np_colors, process=False)
        meshes.append(m)
        np_segms = np.ones(len(np_vertices)) * i
        segms = np_segms if segms is None else np.concatenate((segms, np_segms), axis=0)
    
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_normals=face_normals, vertex_normals=vertex_normals, vertex_colors=colors, process=False)
    # pdb.set_trace()
    mesh = trimesh.util.concatenate(meshes)
    samples, fid = mesh.sample(num_points*2, return_index=True)

    bary = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[fid], points=samples)
    normals = trimesh.unitize((mesh.vertex_normals[mesh.faces[fid]] * trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))
    samples_segms = (segms[mesh.faces[fid]].reshape((-1, 3, 1)) * bary.reshape((-1, 3, 1))).sum(axis=1).reshape(-1).round()
    pcd = torch.from_numpy(samples.reshape((1, samples.shape[0], samples.shape[1])))
    point_idx = farthest_point_sampler(pcd, num_points)[0].cpu().numpy()

    samples = samples[point_idx]
    normals = normals[point_idx]
    samples_segms = samples_segms[point_idx].astype(int)
    
    if DEBUG:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(samples))
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.colors = o3d.utility.Vector3dVector(COLOR20[samples_segms % COLOR20.shape[0]] / 255.)

        o3d.io.write_point_cloud('test.ply', pcd)

    tmp_part_info = copy.deepcopy(part_info)
    for key, val in tmp_part_info.items():
        if val < 0:
            for k, v in tmp_part_info.items():
                if v >= 0 and k.split('_')[-2] == key.split('_')[1]:
                    part_info[key] = v


    return samples, normals, samples_segms, part_info

def parse_motions(motion_path):
    with open(str(motion_path), 'r') as f:
        motion_tree = json.load(f)

    def children(node):
        return node['children']

    nodes= []
    for node in treet.traverse(motion_tree, children, mode='inorder'):
        if node['motion_type'] != 'none':
            tmp_node = {
                'dof_name': node['dof_name'],
                'motion_type': node['motion_type'],
                'center': node['center'],
                'direction': node['direction']
            }
            nodes.append(tmp_node)
    return nodes

def generate_gt(mesh_path, motion_path):
    points, normals, segms, part_info = load_mesh(mesh_path)
    nodes = parse_motions(motion_path)

    assert len(part_info) == len(nodes)+1

    instance_data = {
        'inputs_all': np.column_stack((points, normals)),
        'core_position': np.zeros(num_points),
        'motion_direction_class': np.zeros(num_points),
        'motion_direction_delta': np.zeros((num_points, 3)),
        'motion_position_param': np.zeros((num_points, 3)),
        'motion_dof_type': np.zeros(num_points),
        'all_direction_kmeans': all_direction_kmeans,
        'dof_matrix': np.zeros((len(nodes), 7)),
        'proposal': np.zeros((len(part_info), num_points)),
        'similar_matrix': np.zeros((num_points, num_points))
    }
    for i, node in enumerate(nodes):
        motion_type = 1 if node['motion_type'] == 'rotation' else 2
        center = np.asarray(node['center'])
        direction = np.asarray(node['direction'])
        direction = direction / np.linalg.norm(direction)

        center_disp = points - center
        distance = np.linalg.norm(np.cross(center_disp, direction), axis=1)
        anchor_indices = np.argsort(distance)[:30]
        instance_data['core_position'][anchor_indices] = 1

        rad = np.arccos(np.clip(np.dot(all_direction_kmeans, direction), -1.0, 1.0))
        direction_class = np.argmin(rad)
        direction_delta = direction - all_direction_kmeans[direction_class]
        position_param = center_disp.dot(direction).reshape(-1, 1) * direction - center_disp
        instance_data['motion_direction_class'][anchor_indices] = direction_class + 1
        instance_data['motion_direction_delta'][anchor_indices] = direction_delta
        instance_data['motion_position_param'][anchor_indices] = position_param[anchor_indices]
        tmp = instance_data['motion_dof_type'][anchor_indices]
        tmp[instance_data['motion_dof_type'][anchor_indices] != motion_type] += motion_type
        instance_data['motion_dof_type'][anchor_indices] = tmp
        instance_data['dof_matrix'][i] = np.concatenate((center, direction, [motion_type]))
        segm_id = part_info[node['dof_name']]
        instance_data['proposal'][i+1] = segms == segm_id
    # pdb.set_trace()
    instance_data['proposal'][0] = segms == part_info['none_motion']

    for i in range(num_points):
        instance_data['similar_matrix'][i] = segms == segms[i]

    instance_data['core_position'] = instance_data['core_position'].astype(bool)
    instance_data['proposal'] = instance_data['proposal'].astype(bool)
    instance_data['similar_matrix'] = instance_data['similar_matrix'].astype(bool)
    
    return instance_data

def export_data(batch, output):
    batch_gt = []
    for i, row in batch.iterrows():
        # row = list(batch.iterrows())[0][1]
        cat, inst = row['category'], row['instanceId']
        print(f'{cat}/{inst}')
        mesh_path = Path(dataset_path) / cat / inst / 'part_objs'
        motion_path = Path(dataset_path) / cat / inst / 'motion_attributes.json'
        instance_data = generate_gt(mesh_path, motion_path)
        batch_gt.append([instance_data])
    output_mat = {'Training_data': batch_gt}
    # with open(str(output), 'w+') as f:
    #     hdf5storage.savemat(str(output), output_mat, format='7.3')
    sio.savemat(str(output), output_mat)
    print(f'export to {output}')

def generate_data(df, set='train', categories=[]):
    data_df = df.loc[set]
    if len(categories) > 0:
        data_df = data_df[data_df['category'].isin(categories)]
    
    print(len(data_df))
    B = 16
    if set == 'test':
        B = 1
    count = 0
    # (Path(result_path) / f'{set}_data').mkdir(parents=True, exist_ok=True)
    io.make_clean_folder(Path(result_path) / f'{set}_data')
    for i in range(0, len(data_df), B):
        batch = data_df[i:i+B]
        output_path = Path(result_path) / f'{set}_data' / f'{set}ing_data_{count+1}.mat'
        export_data(batch, output_path)
        count += 1
        

if __name__ == '__main__':
    data_df = split_data()

    generate_data(data_df, 'train', categories=['bike', 'laptop', 'fan', 'car'])
    # generate_data(data_df, 'train', categories=['cabinet'])
    # generate_data(data_df, 'test', categories=['motorbike', 'eyeglasses'])
