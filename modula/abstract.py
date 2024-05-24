import copy
import torch 

from modula.vector import Vector

class Module:
    def __init__(self):
        self.mass = None
        self.sensitivity = None
        self.length = None
        self.children = []
        
    def forward(self, x, w):
        raise NotImplementedError

    def initialize(self, device, dtype):
        raise NotImplementedError

    def normalize(self, w, target_norm):
        raise NotImplementedError

    def regularize(self, w, strength):
        raise NotImplementedError

    def tare(self, absolute=1, relative=None):
        if relative is not None:
            self.mass *= relative
            for child in self.children:
                child.tare(relative = relative)
        else:
            self.tare(relative = absolute / self.mass)

    def print_submodules(self):
        for child in self.children:
            child.print_submodules()

    def __str__(self):
        return f"Module of mass {self.mass} and sensitivity {self.sensitivity}."

    def __call__(self, x, w):
        return self.forward(x, w)

    def __matmul__(self, other):
        if isinstance(other, tuple): other = TupleModule(other)
        return CompositeModule(self, other)

    def __rmatmul__(self, other):
        if isinstance(other, tuple): other = TupleModule(other)
        return other @ self

    def __add__(self, other):
        return Add() @ (self, other)

    def __mul__(self, other):
        assert other != 0, "cannot multiply a module by zero"
        return self @ ScalarMultiply(other)

    def __rmul__(self, other):
        assert other != 0, "cannot multiply a module by zero"
        return ScalarMultiply(other) @ self

    def __truediv__(self, other):
        assert other != 0, "cannot divide a module by zero"
        return self * (1/other)

    def __pow__(self, other):
        assert other >= 0 and other % 1 == 0, "nonnegative integer powers only"
        if other > 0:
            return copy.deepcopy(self) @ self ** (other - 1)
        else:
            return ScalarMultiply(1.0)


class CompositeModule(Module):
    def __init__(self, m1, m0):
        super().__init__()
        self.children = (m0, m1)
        self.length = m0.length + m1.length
        self.mass = m0.mass + m1.mass
        self.sensitivity = m1.sensitivity * m0.sensitivity
        
    def forward(self, x, w):
        m0, m1 = self.children
        w0 = w[:m0.length]
        w1 = w[m0.length:]
        return m1.forward(m0.forward(x, w0), w1)

    def initialize(self, device, dtype=torch.float32):
        m0, m1 = self.children
        return m0.initialize(device, dtype=dtype) & m1.initialize(device, dtype=dtype)

    def normalize(self, w, target_norm=1):
        if self.mass > 0:
            m0, m1 = self.children
            w0 = Vector(w[:m0.length])
            w1 = Vector(w[m0.length:])
            m0.normalize(w0, target_norm=m0.mass / self.mass * target_norm / m1.sensitivity)
            m1.normalize(w1, target_norm=m1.mass / self.mass * target_norm)
        else:
            w *= 0

    def regularize(self, w, strength):
        if self.mass > 0:
            m0, m1 = self.children
            w0 = Vector(w[:m0.length])
            w1 = Vector(w[m0.length:])
            m0.regularize(w0, strength=m0.mass / self.mass * strength / m1.sensitivity)
            m1.regularize(w1, strength=m1.mass / self.mass * strength)


class TupleModule(Module):
    def __init__(self, tuple_of_modules):
        super().__init__()
        self.children = tuple_of_modules
        self.length      = sum(child.length      for child in self.children)
        self.mass        = sum(child.mass        for child in self.children)
        self.sensitivity = sum(child.sensitivity for child in self.children)
        
    def forward(self, x, w):
        output = []
        for child in self.children:
            w_child = w[:child.length]
            output.append(child.forward(x, w_child))
            w = w[child.length:]
        return output

    def initialize(self, device, dtype=torch.float32):
        vector = Vector()
        for child in self.children:
            vector &= child.initialize(device, dtype=dtype)
        return vector

    def normalize(self, w, target_norm=1):
        if self.mass > 0:
            for child in self.children:
                w_child = Vector(w[:child.length])
                child.normalize(w_child, target_norm=child.mass / self.mass * target_norm)
                w = Vector(w[child.length:])
        else:
            w *= 0

    def regularize(self, w, strength):
        if self.mass > 0:
            for child in self.children:
                w_child = Vector(w[:child.length])
                child.regularize(w_child, strength=child.mass / self.mass * strength)
                w = Vector(w[child.length:])


class ScalarMultiply(Module):
    def __init__(self, alpha):
        super().__init__()
        self.mass = 0
        self.sensitivity = abs(alpha)
        self.length = 0
        self.initialize = lambda device, dtype : Vector()
        self.normalize  = lambda w, target_norm : None
        self.regularize = lambda w, strength : None
        self.alpha = alpha

    def forward(self, x, _):
        if isinstance(x, list):
            return [self.forward(xi, _) for xi in x]
        else:
            return self.alpha * x


class Add(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.length = 0
        self.initialize = lambda device, dtype : Vector()
        self.normalize  = lambda w, target_norm : None
        self.regularize = lambda w, strength : None
        self.forward    = lambda x, w : sum(x)
