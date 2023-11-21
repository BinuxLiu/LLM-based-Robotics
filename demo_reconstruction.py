import cv2
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

    walk_v = walk_v + [200, 0]
    action = GrabSim_pb2.Action(scene = scene_id, action = GrabSim_pb2.Action.ActionType.WalkTo, values = walk_v)
    scene = sim_client.Do(action)

def get_camera(part, scene_id=0):
    action = GrabSim_pb2.CameraList(cameras=part, scene=scene_id)
    return sim_client.Capture(action)

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

def transform_point_cloud_y_negative(pcd):
    # Flip the y coordinate of all points in the point cloud
    for point in pcd.points:
        point[1] = -point[1]  # Flip the y coordinate

    return pcd


def T_robot(pose_r2w):
    T = np.array([[1, 0, 0, pose_r2w[0]],
              [0, 1, 0, pose_r2w[1]],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

    Rx = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

    Ry = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

    Rz = np.array([[np.cos(np.radians(pose_r2w[2])), -np.sin(np.radians(pose_r2w[2])), 0, 0],
                [np.sin(np.radians(pose_r2w[2])), np.cos(np.radians(pose_r2w[2])), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
    
    R = np.dot(Rz, np.dot(Ry, Rx))

    T_robot = np.dot(T, R)

    return T_robot

def remove_outliers(point_cloud, nb_neighbors=20, std_ratio=2.0):

    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                     std_ratio=std_ratio)
    return cl


if __name__ == '__main__':
    map_id = 11
    scene_num = 1

    camera_intrinsics = {
        'focal_length': (377.982666015625, 377.2092590332031),
        'principal_point': (319.29327392578125, 242.5348663330078),
        'scale_factor': 0.01
    }

    pose_c2r = np.array([[1.2246468525851679e-16, -0.20791125297546387, 0.9781476259231567, 12.623794555664062],
                        [-1.0, 1.9658280776545684e-16, 3.4183311605647773e-16, -8.605843504483346e-08],
                        [-4.440892098500626e-16, -0.9781476259231567, -0.20791125297546387, 153.7626953125],
                        [0.0, 0.0, 0.0, 1.0]])

    Init()
    SetWorld(map_id, scene_num)
    time.sleep(5.0)


    poses_r2w = [[247,700,45],
                 [500,1300,90], [400,1250,225], [270, 1200,180], [250,1150,225], [247,1100,45], [100,1000,315],[-20, 700, 180], [-20, 750, 0],
                 [0, 900, 270], [70, 880,  315],[0, 720, 270],[0, 650, 180],[0, 550, 270],[0, 450, 180],[0, 350, 270],[0, 250, 180], [0,150, 270], 
                 [150, -200, 315], [260, 0, 90], [260, 200, 90], [260, 400, 90]]


    unified_point_cloud = o3d.geometry.PointCloud()

    for i, pose_r2w in enumerate(poses_r2w):
        T = T_robot(pose_r2w)
        navigation_move(0, 0, pose_r2w)
        time.sleep(1.0)
        img_data = get_camera([GrabSim_pb2.CameraName.Head_Color], 0)
        depth_data = get_camera([GrabSim_pb2.CameraName.Head_Depth], 0)
        color = get_color(img_data)
        depth = get_depth(depth_data)
        point_cloud = create_point_cloud(depth, color, camera_intrinsics)
        point_cloud.transform(pose_c2r)
        point_cloud = transform_point_cloud_y_negative(point_cloud)
        point_cloud.transform(T)
        unified_point_cloud += point_cloud
    unified_point_cloud = remove_outliers(unified_point_cloud)
    

    o3d.visualization.draw_geometries([unified_point_cloud])




