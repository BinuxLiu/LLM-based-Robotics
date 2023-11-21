import asone
import cv2
import sys
import os
import time
import grpc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from asone import utils
from asone import ASOne
from asone.utils import get_names
import matplotlib.pyplot as plt

sys.path.append('./')
sys.path.append('../')

import GrabSim_pb2_grpc
import GrabSim_pb2

detector = ASOne(detector=asone.YOLOV7_PYTORCH, use_cuda=True) # Set use_cuda to False for cpu
filter_classes = None # Set to None to detect all classes
names = get_names()

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

def get_camera(part, scene_id=0):
    action = GrabSim_pb2.CameraList(cameras=part, scene=scene_id)
    return sim_client.Capture(action)

def detect(img_data, save_path = None):
    im = img_data.images[0]
    frame = np.frombuffer(im.data, dtype=im.dtype).reshape((im.height, im.width, im.channels))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    dets, img_info = detector.detect(frame, filter_classes=filter_classes)
    bbox_xyxy = dets[:, :4]
    scores = dets[:, 4]
    class_ids = dets[:, 5]
    frame = utils.draw_boxes(frame, bbox_xyxy, class_ids=class_ids)

    if save_path:
        cv2.imwrite(save_path, frame)

    return bbox_xyxy, class_ids

def get_depth(img_data):
    im = img_data.images[0]
    d = np.frombuffer(im.data, dtype=im.dtype).reshape((im.height, im.width, im.channels))
    return d

def compute_object_pose(pose_c2r, pixel_x, pixel_y, depth, camera_intrinsics, pose_r2w):
    fx, fy = camera_intrinsics['focal_length']
    cx, cy = camera_intrinsics['principal_point']
    normalized_x = (pixel_x - cx) / fx
    normalized_y = (pixel_y - cy) / fy

    camera_x = normalized_x * depth
    camera_y = normalized_y * depth
    camera_z = depth

    camera_coordinates = np.array([camera_x, camera_y, camera_z, 1], dtype=object)


    robot_coordinates = np.dot(pose_c2r, camera_coordinates)[:3]    
    robot_homogeneous_coordinates = np.array([robot_coordinates[0], -robot_coordinates[1], robot_coordinates[2], 1], dtype=object)

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

    world_coordinates = np.dot(T_robot, robot_homogeneous_coordinates)[:3]
    return  [float(coord[0]) for coord in world_coordinates]



if __name__ == '__main__':
    map_id = 11
    scene_num = 1

    camera_intrinsics = {
        'focal_length': (377.982666015625, 377.2092590332031),
        'principal_point': (319.29327392578125, 242.5348663330078),
        'scale_factor': 1
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
    objects = []


    os.makedirs('images', exist_ok=True)
    
    for s, pose_r2w in enumerate(poses_r2w):
        navigation_move(0, 0, pose_r2w)
        img_data = get_camera([GrabSim_pb2.CameraName.Head_Color], 0)
        depth_data = get_camera([GrabSim_pb2.CameraName.Head_Depth], 0)
        bounding_boxes, class_ids = detect(img_data)
        centers_x = (bounding_boxes[:, 0] + bounding_boxes[:, 2]) / 2
        centers_y = (bounding_boxes[:, 1] + bounding_boxes[:, 3]) / 2
        depth_img = get_depth(depth_data)
        for i in range(len(bounding_boxes)):
            obj_name = names[int(class_ids[i])]
            depth_value = depth_img[int(centers_y[i]), int(centers_x[i])]
            if depth_value < 150:
                position = compute_object_pose(pose_c2r, centers_x[i], centers_y[i], depth_value, camera_intrinsics, pose_r2w)
                objects.append((position, obj_name, pose_r2w))


    n_poses = len(poses_r2w)
    color_gradient = [plt.cm.jet(i / n_poses) for i in range(n_poses)]

    fig, ax = plt.subplots()

    for i, pose in enumerate(poses_r2w):
        ax.scatter(pose[0], pose[1], marker='o', color=color_gradient[i], label=f'Pose {pose}', s=100)

    # Connecting the poses in order
    for i in range(len(poses_r2w) - 1):
        ax.plot([poses_r2w[i][0], poses_r2w[i+1][0]], [poses_r2w[i][1], poses_r2w[i+1][1]], color='k', linestyle='-', linewidth=2)

    for obj in objects:
        coords, obj_type, pose_r2w = obj
        pose_index = poses_r2w.index(pose_r2w)
        ax.scatter(coords[0], coords[1], label=obj_type, color=color_gradient[pose_index], marker='x',)
        ax.plot([coords[0], pose_r2w[0]], [coords[1], pose_r2w[1]], linestyle='--', color=color_gradient[pose_index])


    ax.set_title('2D Visualization of Object Positions')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # ax.legend()

    plt.savefig('./demo/2D_object_graph.png')


