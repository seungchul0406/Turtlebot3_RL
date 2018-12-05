from numpy import zeros
from numpy.random import randn

class OUNoise:
    """ docstring for OUNoise """
    def __init__(self,action_dimension,mu=0, theta=0.05, sigma=0.02):
        self.action_dimension=action_dimension
        self.mu=mu
        self.theta=theta
        self.sigma=sigma
        self.state=zeros(self.action_dimension)
        self.reset()

    def reset(self):
        self.state=zeros(self.action_dimension)

    def noise(self):
        x=self.state
        dx=self.theta*(self.mu-x)+self.sigma*randn(len(x))
        self.state=x+dx
        return self.state.reshape([1,self.action_dimension])
