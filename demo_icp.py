import cv2
import os
import sys
import time
import grpc
import numpy as np
import open3d as o3d

sys.path.append('./')
sys.path.append('../')

import GrabSim_pb2_grpc
import GrabSim_pb2


channel = grpc.insecure_channel('localhost:30001',options=[
            ('grpc.max_send_message_length', 1024*1024*1024),
            ('grpc.max_receive_message_length', 1024*1024*1024)
        ])

sim_client = GrabSim_pb2_grpc.GrabSimStub(channel)

def Init():
    sim_client.Init(GrabSim_pb2.NUL())

def SetWorld(map_id = 0, scene_num = 1):
    print('------------------SetWorld----------------------')
    world = sim_client.SetWorld(GrabSim_pb2.BatchMap(count = scene_num, mapID = map_id))

def navigation_move(scene_id=0, map_id=0, walk_v = [247, 520, 180]):
    print('------------------navigation_move----------------------')
    scene = sim_client.Observe(GrabSim_pb2.SceneID(value=scene_id))

    walk_v = walk_v + [-1, 0]
    action = GrabSim_pb2.Action(scene = scene_id, action = GrabSim_pb2.Action.ActionType.WalkTo, values = walk_v)
    scene = sim_client.Do(action)


def create_point_cloud(depth_image, color_image, camera_intrinsics):
    height, width = depth_image.shape[:2]
    fx, fy = camera_intrinsics['focal_length']
    cx, cy = camera_intrinsics['principal_point']
    scale_factor = camera_intrinsics['scale_factor']

    points = []
    colors = []

    for v in range(height):
        for u in range(width):

            z = depth_image[v, u]
            if z < 500: 
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                color = color_image[v, u] / 255.0

                points.append([x, y, z])
                colors.append(color)

    points = np.array(points, dtype=np.float64)
    colors = np.array(colors, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def apply_icp(source_pcd, target_pcd, initial_transform=np.eye(4), threshold=0.02):
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return icp_result.transformation


def transform_point_cloud(pcd, transform):
    pcd_transformed = pcd.transform(transform)
    return pcd_transformed


def get_color(img_data, save_path = None):
    im = img_data.images[0]
    print()
    color = np.frombuffer(im.data, dtype=im.dtype).reshape((im.height, im.width, im.channels))
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    return color

def get_depth(img_data, save_path=None):
    im = img_data.images[0]
    depth = np.frombuffer(im.data, dtype=im.dtype).reshape((im.height, im.width, im.channels))

    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]

    depth = np.nan_to_num(depth)

    depth = depth.astype(np.float32)

    if save_path:
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_8bit = np.uint8(depth_normalized)
        cv2.imwrite(save_path, depth_8bit)

    return depth

def get_camera(part, scene_id=0):
    action = GrabSim_pb2.CameraList(cameras=part, scene=scene_id)
    return sim_client.Capture(action)

if __name__ == '__main__':

    map_id = 11
    scene_num = 1

    camera_intrinsics = {
        'focal_length': (377.982666015625, 377.2092590332031),
        'principal_point': (319.29327392578125, 242.5348663330078),
        'scale_factor': 1
    }


    Init()
    SetWorld(map_id, scene_num)
    time.sleep(5.0)

    poses_r2w = [[0, 650, 180],[0, 450, 180] ,[0, 350, 120] ,[0, 250, 60] ]

    os.makedirs('images', exist_ok=True)
    
    previous_point_cloud = o3d.geometry.PointCloud()

    for s, pose_r2w in enumerate(poses_r2w):
        navigation_move(0, 0, pose_r2w)
        time.sleep(1.0)
        img_data = get_camera([GrabSim_pb2.CameraName.Head_Color], 0)
        depth_data = get_camera([GrabSim_pb2.CameraName.Head_Depth], 0)
        color = get_color(img_data)
        depth = get_depth(depth_data)
        current_point_cloud = create_point_cloud(depth, color, camera_intrinsics)
        if s != 0:
            icp_transform = apply_icp(previous_point_cloud, current_point_cloud)
            print("当前帧与上一帧的相对位姿：")
            print(icp_transform)
        previous_point_cloud = current_point_cloud

    
