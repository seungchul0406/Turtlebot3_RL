import tensorflow as tf
from numpy import reshape,tanh,random
from agent_modules.build_actor_critic import Build_network
from agent_modules.additional_functions import l2_regularizer,gradient_inverter
from agent_modules.ou_noise import OUNoise


class DDPG(object):

    def __init__(self,config):
        if config.gpu:
            sess_config=tf.ConfigProto()
            # sess_config.gpu_options.allow_growth=True
            sess_config.gpu_options.per_process_gpu_memory_fraction = 0.1
        else:
            sess_config=None
        self.state_dim=config.state_dim
        self.action_dim=config.action_dim
        self.gamma=tf.constant(config.gamma,dtype=tf.float32,name='gamma')
        self.sess=tf.Session(config=sess_config)
        self.var_init=tf.global_variables_initializer()
        self.reward=tf.placeholder(tf.float32,[None,1])
        self.done=tf.placeholder(tf.float32,[None,1])
        self.target_q=tf.placeholder(tf.float32,[None,1])
        # self.noise=tf.placeholder(tf.float32,[None,config.action_dim])
        # build network
        self.actor_net=Build_network(self.sess,config,'actor_net')
        self.actor_target=Build_network(self.sess,config,'actor_target')
        self.critic_net=Build_network(self.sess,config,'critic_net')
        self.critic_target=Build_network(self.sess,config,'critic_target')
        # update critic
        y=self.reward+tf.multiply(self.gamma,tf.multiply(self.target_q,1.0-self.done))
        # y=self.reward+tf.multiply(self.gamma,self.target_q)
        q_loss=tf.reduce_sum(tf.pow(self.critic_net.out_-y,2))/config.batch_size+ \
            config.l2_penalty*l2_regularizer(self.critic_net.var_list)
        self.update_critic=tf.train.AdamOptimizer( \
            learning_rate=config.critic_learning_rate).minimize(q_loss,var_list=self.critic_net.var_list)
        # update actor
        act_grad_v=tf.gradients(self.critic_net.out_,self.critic_net.action)
        action_gradients=[act_grad_v[0]/tf.to_float(tf.shape(act_grad_v[0])[0])]
        del_Q_a=gradient_inverter( \
            config.action_bounds,action_gradients,self.actor_net.out_)
        parameters_gradients=tf.gradients(
            self.actor_net.out_,self.actor_net.var_list,-del_Q_a)
        self.update_actor=tf.train.AdamOptimizer( \
            learning_rate=config.actor_learning_rate) \
            .apply_gradients(zip(parameters_gradients,self.actor_net.var_list))
        # target copy
        self.assign_target= \
            [self.actor_target.variables[var].assign( \
                self.actor_net.variables[var.replace('_target','_net')] \
            ) for var in self.actor_target.variables.keys()]+ \
            [self.critic_target.variables[var].assign( \
                self.critic_net.variables[var.replace('_target','_net')] \
            ) for var in self.critic_target.variables.keys()]
        self.assign_target_soft= \
            [self.actor_target.variables[var].assign( \
                config.tau*self.actor_net.variables[var.replace('_target','_net')]+ \
                (1-config.tau)*self.actor_target.variables[var] \
            ) for var in self.actor_target.variables.keys()]+ \
            [self.critic_target.variables[var].assign( \
                config.tau*self.critic_net.variables[var.replace('_target','_net')]+ \
                (1-config.tau)*self.critic_target.variables[var] \
            ) for var in self.critic_target.variables.keys()]
        # initialize variables
        self.var_init=tf.global_variables_initializer()
        self.sess.run(self.var_init)
        self.sess.run(self.assign_target)
        # self.ou=OUNoise(config.action_dim)
        self.a_scale,self.a_mean=self.sess.run(
            [self.actor_net.a_scale,self.actor_net.a_mean])

    def policy(self,state,epsilon=1.0):
        action=self.sess.run(self.actor_net.out_before_activation, \
            feed_dict={self.actor_net.state:state})
        action=self.a_scale*tanh(action+0.4*random.randn(1,2)*epsilon)+self.a_mean
        return action
    
    def reset(self):
        self.sess.run(self.var_init)

    def update(self,batch):
        state0=reshape(batch['state0'],[-1,self.state_dim])
        state1=reshape(batch['state1'],[-1,self.state_dim])
        action0=reshape(batch['action0'],[-1,self.action_dim])
        reward=reshape(batch['reward'],[-1,1])
        done=reshape(batch['done'],[-1,1])
        target_action=self.actor_target.evaluate(state1)
        target_q=self.critic_target.evaluate(state1,action=target_action)
        self.sess.run(self.update_critic, \
                      feed_dict={self.critic_net.state:state0, \
                                 self.critic_net.action:action0, \
                                 self.reward:reward, \
                                 self.target_q:target_q, \
                                 self.done:done})
        self.sess.run(self.update_actor, \
                      feed_dict={self.critic_net.state:batch['state0'], \
                                 self.critic_net.action:batch['action0'], \
                                 self.actor_net.state:batch['state0']})
        self.sess.run(self.assign_target_soft)
    
    def load(self,saved_variables):
        self.sess.run( \
            [self.actor_net.variables[var].assign(saved_variables[var]) \
                for var in self.actor_net.variables.keys()]+ \
            [self.critic_net.variables[var].assign(saved_variables[var]) \
                for var in self.critic_net.variables.keys()]+ \
            self.assign_target)
    
    def return_variables(self):
        return dict({name:self.sess.run(name) \
                    for name in self.actor_net.variables.keys()}, \
               **{name:self.sess.run(name) \
                    for name in self.critic_net.variables.keys()})
