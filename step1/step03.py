import numpy as np 
from abc import ABC, abstractmethod

class Variable:

    def __init__(self, data: np.ndarray):
        self.data = data 

    def __repr__(self):
        return f"<{self.__class__.__name__}(ndim={self.data.ndim})>"


class Function(ABC):

    def __call__(self, input: Variable):
        x = input.data 
        y = self.forward(x)
        o = Variable(data=y)
        return o 
    
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError() 
    
class Square(Function):

    def forward(self, x):
        return x ** 2
    
class Exp(Function):

    def forward(self, x):
        return np.exp(x) 
    
x = Variable(np.array(0.5))
square = Square()
exp = Exp()

a = square(x)
b = exp(a)
c = square(b)

print(c.data)
