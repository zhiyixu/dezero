import numpy as np 

class Variable:

    def __init__(self, data: np.ndarray):
        self.data = data 

    def __repr__(self):
        return f"<{self.__class__.__name__}(ndim={self.data.ndim})>"


data = np.array(1.0)

print(data.ndim)

x = Variable(data=data)
print(x.data)
print(x)