from __future__ import print_function
import os,sys,time
from numpy import reshape,save,load
from configuration import config

from agent.ddpg import DDPG
from environment.turtlebot3_obstacles import Turtlebot3_obstacles

# config.autolaunch=False
config.reward_param=0.8
# env=Turtlebot_obstacles(config)
# agent=DDPG(config)

# env.launch()

def train(port):    
    config.api_port=port
    env=Turtlebot3_obstacles(config,port)

    agent=DDPG(config)
    # agent.load(load('savedir/weight_0.0.npy').item())

    #params=[0.1303, 0.2576, 0.4564]
    #params=[0.0973,0.1315,0.1305,0.2559,0.3058,0.4629]
    params=[0.4629]
    env.launch()

    for param in params:
        env.epoch=0
        env.reward_param=param
        #agent.load(load('savedir/weight_'+str(param)+'.npy').item())
        for episode in range(config.max_episode):
            env.reset()
            print('Rho:',param,'\tEpisode:',episode+1)
            env.start()
            state,done=env.step([0,0])
            for step in range(config.max_step):
                epsilon=0.99998**env.epoch
                action=agent.policy(reshape(state,[1,config.state_dim]),epsilon=epsilon)
                state,done=env.step(reshape(action,[config.action_dim]))
                if env.replay.buffersize>200:
                    batch=env.replay.batch()            
                    agent.update(batch)
                if done==1:
                    break
            if step>=config.max_step-1:
                print(' | Timeout')
            if (episode+1)%100==0:
                save(os.path.join( \
                    'savedir','weight_'+str(param)+'.npy'), \
                    agent.return_variables())
            if env.epoch>config.max_epoch:
                break

def test(port):
    config.api_port=port
    env=Turtlebot3_obstacles(config,port)
    agent=DDPG(config)
    #params=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    #params=[0.1303, 0.2576, 0.4564]
    params=[0.0973,0.1315,0.1305,0.2559,0.3058,0.4629]
    trajs={str(param):[] for param in params}

    env.launch()

    for param in params:
        env.reward_param=param
        agent.load(load('savedir/weight_'+str(param)+'.npy').item())
        env.reset()
        env.start()
        state,done=env.step([0,0])
        traj=[]
        for step in range(config.max_step):
            action=agent.policy(reshape(state,[1,config.state_dim]),epsilon=0.0)
            state,obs,done=env.step(reshape(action,[config.action_dim]),return_obs=True)
            traj.append(obs)
            if done==1:
                break
        if step>=config.max_step-1:
            print(' | Timeout')
        trajs[str(param)]=traj
        save('recovered_traj2.npy',trajs)

if __name__=='__main__':
	#train(20000)
    test(20000)
