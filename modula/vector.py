import torch


class Vector:
    """For doing algebra on lists of tensors.

    An instance of Vector stores a list of tensors. Vectors can be
    added, subtracted, scalar-multiplied, elementwise-multiplied, etc.
    We also support in-place operations for efficiency.

    Vectors are intended to store the weights of a neural net,
    allowing weight updates to be implemented using simple algebra.
    """

    def __init__(self, tensor_or_tensor_list = []):
        """Stores a list of tensors."""
        if isinstance(tensor_or_tensor_list, torch.Tensor):
            self.tensor_list = [tensor_or_tensor_list]
        elif isinstance(tensor_or_tensor_list, list):
            self.tensor_list = tensor_or_tensor_list
        elif isinstance(tensor_or_tensor_list, tuple):
            self.tensor_list = tensor_or_tensor_list
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        """Allows Vectors to be indexed and looped over."""
        return self.tensor_list[item]

    def __len__(self):
        return len(self.tensor_list)

    def grad(self):
        """Returns the gradient list of this Vector."""
        return Vector([tensor.grad for tensor in self])

    def zero_grad(self):
        """Delete the gradients of this Vector."""
        for tensor in self:
            tensor.grad = None

    def zero_nans(self):
        """Set any nans or infs to zero, in-place."""
        for tensor in self:
            tensor.nan_to_num_(0,0,0)

    @torch.no_grad()
    def all_reduce(self):
        """Sums this vector over all workers"""
        for tensor in self:
            torch.distributed.all_reduce(tensor, torch.distributed.ReduceOp.SUM)

    @torch.no_grad()
    def broadcast(self):
        """Broadcasts this vector from worker zero to all other workers."""
        for tensor in self:
            torch.distributed.broadcast(tensor, src=0)

    def __str__(self):
        """Lets us print the Vector."""
        return str([t for t in self])

    def __and__(self, other):
        """Conatenate two Vectors."""
        return Vector(self.tensor_list + other.tensor_list)

    def __iadd__(self, other):
        """In-place add."""
        if len(self) == 0: return self
        if isinstance(other, Vector): other = other.tensor_list
        torch._foreach_add_(self.tensor_list, other)
        return self

    def __add__(self, other):
        """Add."""
        if len(self) == 0: return Vector()
        if isinstance(other, Vector): other = other.tensor_list
        new_list = torch._foreach_add(self.tensor_list, other)
        return Vector(new_list)

    def __mul__(self, other):
        """Multiply."""
        if len(self) == 0: return Vector()
        if isinstance(other, Vector): other = other.tensor_list
        new_list = torch._foreach_mul(self.tensor_list, other)
        return Vector(new_list)

    def __rmul__(self, other):
        """Multiply from the left."""
        return self * other

    def __imul__(self, other):
        """In-place multiply."""
        if len(self) == 0: return self
        if isinstance(other, Vector): other = other.tensor_list
        torch._foreach_mul_(self.tensor_list, other)
        return self

    def __isub__(self, other):
        """In-place subtract."""
        if len(self) == 0: return self
        if isinstance(other, Vector): other = other.tensor_list
        torch._foreach_sub_(self.tensor_list, other)
        return self

    def __sub__(self, other):
        """Subtract."""
        if len(self) == 0: return Vector()
        if isinstance(other, Vector): other = other.tensor_list
        new_list = torch._foreach_sub(self.tensor_list, other)
        return Vector(new_list)

    def __itruediv__(self, other):
        """In-place division."""
        if len(self) == 0: return self
        if isinstance(other, Vector): other = other.tensor_list
        torch._foreach_div_(self.tensor_list, other)
        return self

    def __truediv__(self, other):
        """Division."""
        if len(self) == 0: return Vector()
        if isinstance(other, Vector): other = other.tensor_list
        new_list = torch._foreach_div(self.tensor_list, other)
        return Vector(new_list)

    def __ipow__(self, other):
        """In-place power."""
        if len(self) == 0: return self
        if isinstance(other, Vector): other = other.tensor_list
        torch._foreach_pow_(self.tensor_list, other)
        return self

    def __pow__(self, other):
        """Power."""
        if len(self) == 0: return Vector()
        if isinstance(other, Vector): other = other.tensor_list
        new_list = torch._foreach_pow(self.tensor_list, other)
        return Vector(new_list)


if __name__ == "__main__":

    a = Vector([torch.tensor(2.0), torch.tensor(1.0)])

    a *= 2;  print(a)
    a += 1;  print(a)
    a -= 1;  print(a)
    a /= 2;  print(a)
    a **= 2; print(a)

    a = Vector([torch.tensor(2.0), torch.tensor(1.0)])

    a **= a; print(a)
    a *= a;  print(a)
    a /= a;  print(a)
    a += a;  print(a)
    a -= a;  print(a)

    a = Vector([torch.tensor(2.0), torch.tensor(1.0)])

    a = a * 2;  print(a)
    a = a + 1;  print(a)
    a = a - 1;  print(a)
    a = a / 2;  print(a)
    a = a ** 2; print(a)

    a = Vector([torch.tensor(2.0), torch.tensor(1.0)])

    a = a * a;  print(a)
    a = a + a;  print(a)
    a = a / a;  print(a)
    a = a ** a; print(a)
    a = a - a;  print(a)
