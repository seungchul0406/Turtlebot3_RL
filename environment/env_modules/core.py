from __future__ import print_function
import os
import random
import subprocess
import time
import numpy as np

import vrep
from replay import Replay

class Core(object):

    def __init__(self, config, scene):
        self.vrep_path = config.vrep_path
        self.viz = config.visualization
        self.autolaunch = config.autolaunch
        self.port = config.api_port
        self.clientID = None
        self.scene = scene
        self.dt = config.dt
        self.replay = Replay(config.max_buffer, config.batch_size)
        self.batch_size = config.batch_size

    def vrep_launch(self):
        if self.autolaunch:
            if self.viz:
                vrep_exec=self.vrep_path+'/vrep.sh '
                t_val = 5.0
            else:
                vrep_exec=self.vrep_path+'/vrep.sh -h '
                t_val = 1.0
            synch_mode_cmd='-gREMOTEAPISERVERSERVICE_'+str(self.port)+'_FALSE_TRUE '
            subprocess.call(
                vrep_exec+synch_mode_cmd+self.scene+' &',
                shell=True
            )
            time.sleep(t_val)      
        self.clientID=vrep.simxStart(
            '127.0.0.1',
            self.port,
            True,
            True,
            5000,
            5
        )
    
    def vrep_start(self):
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        vrep.simxSynchronous(self.clientID, True)
    
    def vrep_reset(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        time.sleep(0.1)
    
    def pause(self):
        vrep.simxPauseSimulation(
            self.clientID,vrep.simx_opmode_oneshot)
    
    def close(self):
        self.vrep_reset()
        while vrep.simxGetConnectionId(self.clientID) != -1:
            vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxFinish(self.clientID)
        self.replay.clear()

if __name__ == '__main__':
    env = Core(None, None)