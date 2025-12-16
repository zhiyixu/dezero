import numpy as np 
from abc import ABC, abstractmethod
from typing import Union, Any

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
    
def numerical_diff(f: Union[Function, Any], x: Variable, esp=1e-4):
    x0 = Variable(x.data-esp)
    x1 = Variable(x.data+esp)

    y0 = f(x0)
    y1 = f(x1)
    return (y1.data-y0.data)/(2*esp)

x = Variable(np.array(2))
f = Square()
dy = numerical_diff(f=f, x=x)
print(dy)

def fx(x):
    squ = Square()
    exp = Exp()

    return squ(exp(squ(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f=fx, x=x)
print(dy)