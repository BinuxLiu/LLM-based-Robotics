#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#enconding = utf8
import sys
import time
import grpc

sys.path.append('./')
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import GrabSim_pb2_grpc
import GrabSim_pb2

channel = grpc.insecure_channel('localhost:30001',options = [
            ('grpc.max_send_message_length', 1024*1024*1024),
            ('grpc.max_receive_message_length', 1024*1024*1024)
        ])

sim_client = GrabSim_pb2_grpc.GrabSimStub(channel)


def Init():
    sim_client.Init(GrabSim_pb2.NUL())


def AcquireAvailableMaps():
    AvailableMaps = sim_client.AcquireAvailableMaps(GrabSim_pb2.NUL())
    print(AvailableMaps)


def SetWorld(map_id = 0, scene_num = 1):
    print('------------------SetWorld----------------------')
    world = sim_client.SetWorld(GrabSim_pb2.BatchMap(count = scene_num, mapID = map_id))


def Reset(scene_id = 0):
    print('------------------Reset----------------------')
    scene = sim_client.Reset(GrabSim_pb2.ResetParams(scene = scene_id))
    print(scene)


def navigation_move(scene_id=0, map_id=0, walk_v = [247, 520, 180]):
    print('------------------navigation_move----------------------')
    scene = sim_client.Observe(GrabSim_pb2.SceneID(value=scene_id))

    walk_v = walk_v + [-1, 0]
    action = GrabSim_pb2.Action(scene = scene_id, action = GrabSim_pb2.Action.ActionType.WalkTo, values = walk_v)
    scene = sim_client.Do(action)


def rotate_joints(scene_id = 0, action_list = []):
    print('------------------rotate_joints----------------------')

    for values in action_list:
        action = GrabSim_pb2.Action(scene = scene_id, action = GrabSim_pb2.Action.ActionType.RotateJoints, values = values)
        scene = sim_client.Do(action)


def reset_joints(scene_id=0):
    print('------------------reset_joints----------------------')
    values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    action = GrabSim_pb2.Action(scene = scene_id, action = GrabSim_pb2.Action.ActionType.RotateJoints, values = values)
    scene = sim_client.Do(action)



def rotate_fingers(scene_id=0):
    print('------------------rotate_fingers----------------------')
    values = [-6, 0, 45, 45, 45, -6, 0, 45, 45, 45]
    action = GrabSim_pb2.Action(scene = scene_id, action = GrabSim_pb2.Action.ActionType.Finger, values = values)
    scene = sim_client.Do(action)



def reset_fingers(scene_id=0):
    print('------------------reset_fingers----------------------')
    values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    action = GrabSim_pb2.Action(scene = scene_id, action = GrabSim_pb2.Action.ActionType.Finger, values = values)
    scene = sim_client.Do(action)



if __name__ == '__main__':
    map_id = 11
    scene_num = 1
    
    Init()
    AcquireAvailableMaps()
    SetWorld(map_id, scene_num)
    time.sleep(5.0)

    for i in range(scene_num):

        Reset(i)

        navigation_move(i, map_id, walk_v = [249.0, -155.0, 0])
        time.sleep(1.0)
        rotate_joints(i, [[0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 36.0, -39.37, 37.2, -92.4, 4.13, -0.62, 0.4],
                         [0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 36.0, -39.62, 34.75, -94.80, 3.22, -0.26, 0.85],
                         [0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 32.63, -32.80, 15.15, -110.70, 6.86, 2.36, 0.40],
                         [0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 28.18, -27.92, 6.75, -115.02, 9.46, 4.28, 1.35],
                         [0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 4.09, -13.15, -11.97, -107.35, 13.08, 8.58, 3.33]])
        time.sleep(1.0)
        rotate_fingers(i)
        time.sleep(1.0)
        rotate_joints(i, [[0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, -29.09, -20.15, -11.97, -70.35, 13.08, 8.58, 3.33],
                         [0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, -30.09, -19.15, -11.97, -70.35, 13.08, 8.58, 3.33]])
        time.sleep(1.0)
        reset_joints(i)
        time.sleep(1.0)
        reset_fingers(i)
        navigation_move(i, map_id, walk_v = [247, 520, 0])