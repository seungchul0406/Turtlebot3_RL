from __future__ import print_function
import os
import sys
import time
from numpy import array,reshape,linalg,arctan2,pi,expand_dims
from random import choice
from env_modules import vrep
from env_modules.core import Core


scene_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'scenes')


class Turtlebot3_obstacles(Core):

    def __init__(self,config,port):
        Core.__init__(
            self,
            config,
            os.path.join(scene_dir,'turtlebot3.ttt'))
        self.d=0.079
        self.r=0.033
        self.goal_set=[[3.5,3.3],[3.5,3.3]]
        #self.goal_set=[[3.5,3.3],[3.5,3.3],[3.5,3.3],[2.0,1.8],[3.5,0.0],[2.0,3.3]]
        self.reward_param=config.reward_param
        self.action_prev=[0.0,0.0]
        self.state0=None
        self.goal_dist_prev=None
        self.port=config.api_port
    
    def launch(self):
        self.vrep_launch()
        vrep.simxSynchronousTrigger(self.clientID)
        self.joint_handles=[ \
            vrep.simxGetObjectHandle( \
                self.clientID,name,vrep.simx_opmode_blocking)[1] \
                for name in ['wheel_right_joint','wheel_left_joint']]
        self.body_handle=vrep.simxGetObjectHandle(self.clientID, \
            'Turtlebot3',vrep.simx_opmode_blocking)[1]
        self.goal_handle=vrep.simxGetObjectHandle(self.clientID, \
            'Goal',vrep.simx_opmode_blocking)[1]
        self.epoch=0
        self.count=0
    
    def reset(self):
        self.vrep_reset()
        self.goal=choice(self.goal_set)
        vrep.simxSetObjectPosition(self.clientID, \
            self.goal_handle,-1,self.goal+[0],vrep.simx_opmode_blocking)
        self.state0=None
        self.action_prev=[0.0,0.0]
        self.goal_dist_prev=None
        self.epoch+=self.count
        self.count=0
        self.reward_sum=0.0
        time.sleep(0.2)
    
    def start(self):
        self.vrep_start()
        t=vrep.simxGetLastCmdTime(self.clientID)
        vrep.simxSynchronousTrigger(self.clientID)
        self.controller([0.0,0.0])
        while vrep.simxGetLastCmdTime(self.clientID)-t<self.dt:
            lrf_bin=vrep.simxGetStringSignal(self.clientID, \
                'hokuyo_data',vrep.simx_opmode_streaming)[1]
            pose=vrep.simxGetObjectPosition(self.clientID, \
                self.body_handle,-1,vrep.simx_opmode_streaming)[1]
    
    def reward(self,lrf,goal_dist,action):
        # return 10*(self.goal_dist_prev-goal_dist) \
        #        -(1/min(lrf)-1)/5.0 \
        #        -0.5*(1+self.reward_param*action[1]**2)
        r_g=8.0*(self.goal_dist_prev-goal_dist)
        rho=min(lrf)
        if (rho-0.01972)<self.reward_param/5.578:
            r_o=min((1.0/(rho-0.01972)-5.578/self.reward_param)**2,1)
        else:
            r_o=0
        return r_g-r_o

    def step(self,action,return_obs=False):
        self.count+=1
        self.controller(action)
        t=vrep.simxGetLastCmdTime(self.clientID)
        vrep.simxSynchronousTrigger(self.clientID)
        while vrep.simxGetLastCmdTime(self.clientID)-t<self.dt:
            pose=vrep.simxGetObjectPosition(self.clientID, \
                self.body_handle,-1,vrep.simx_opmode_blocking)[1]
            orientation=vrep.simxGetObjectOrientation(self.clientID, \
                self.body_handle,-1,vrep.simx_opmode_blocking)[1][2]
            goal_pos=vrep.simxGetObjectPosition(self.clientID, \
                    self.goal_handle,self.body_handle,vrep.simx_opmode_blocking)[1]
            # vel=vrep.simxGetObjectVelocity(self.clientID, \
            #     self.body_handle,vrep.simx_opmode_streaming)
            lrf_bin=vrep.simxGetStringSignal(self.clientID, \
                'hokuyo_data',vrep.simx_opmode_blocking)[1]
            lrf=array(vrep.simxUnpackFloats(lrf_bin),dtype=float)/5.578
        goal_dist=linalg.norm(goal_pos[0:2])
        goal_angle=arctan2(-goal_pos[0],goal_pos[1])
        state1=list(lrf)+[action[0]/0.26,action[1]/0.8]
        state1+=[goal_dist/5.578,goal_angle/pi] \
                    if goal_dist<5.578 else \
                [1,goal_angle/pi]
        if self.goal_dist_prev!=None:
            reward=self.reward(lrf,goal_dist,action)
            self.goal_dist_prev=goal_dist
            self.reward_sum+=reward
        sys.stderr.write( \
            '\rstep:%d| goal:% 2.1f,% 2.1f | pose:% 2.1f,% 2.1f | avg.reward:% 4.2f' \
            %(self.count,self.goal[0],self.goal[1],pose[0],pose[1],self.reward_sum/self.count))
        if min(lrf)<0.01972:
            done=1
            print(' | Fail')
        elif goal_dist<0.1:
            done=1
            print(' | Success')
        else:
            done=0
        if self.state0!=None:
            self.replay.add({'state0':self.state0, \
                            'action0':action, \
                            'reward':reward, \
                            'state1':state1, \
                            'done':done})
        self.state0=state1
        self.action_prev=action
        self.goal_dist_prev=goal_dist
        if return_obs:
            obs={
                'state':state1,
                'lidar':list(lrf),
                'action':action,
                'pose':[pose[0],pose[1],orientation],
                'goal':[goal_dist,goal_angle],
                'reward':reward,
                'time':t+self.dt
            }
            return state1,obs,done
        else:
            return state1,done
    
    def controller(self,action):
        vel_right=2.0*(action[0]+self.d*action[1])/self.r
        vel_left=2.0*(action[0]-self.d*action[1])/self.r
        vrep.simxSetJointTargetVelocity(self.clientID, \
            self.joint_handles[0],vel_right,vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(self.clientID, \
            self.joint_handles[1],vel_left,vrep.simx_opmode_streaming)
