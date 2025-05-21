import math
import random
class Value:
    def __init__(self,data,_childern=(),_op='',lable='') -> None:
        self.data = data 
        self.prev = set(_childern)
        self.grad = 0
        self.op =_op
        self.lable = lable
        self._backward = lambda:None
    def __repr__(self) -> str:
        return str(self.data)
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,(self,other),'+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __radd__(self,other):
        return self+other
    def __neg__(self):
        return self * -1
    def __sub__(self,other):
        return self + (-other)
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data,(self,other),'*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __rmul__(self,other):
        return self*other
    def __pow__(self,other):
        assert isinstance(other,(int,float))
        out = Value(self.data**other,(self,))
        def _backward():
            self.grad += other*self.data**(other-1) * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self,),'tanh')

        def _backward():
            self.grad += (1-t**2)*out.grad
        out._backward = _backward
        return out
    def exp(self):
        x = self.data
        t = (math.exp(x))
        out = Value(t,(self,),'tanh')

        def _backward(): 
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

class Neuron:

    def __init__(self,nin):
        self.w  = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    def __call__(self,x):
        act = sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        out = act.tanh()
        return out
    def parameters(self):
        return self.w + [self.b]
class Layer:
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
class MLP:
    def __init__(self,nin,nouts):
        sz = [nin]+ nouts
        self.layers =[Layer(sz[i],sz[i+1]) for i in range(len(nouts))]
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]



n = MLP(2,[4,4,1])

xs = [
    [1,1],
    [1,0],
    [0,1],
    [0,0]
]
ys = [0,1,1,0]

epochs = 1000
for i in range(epochs):
    ypred = [n(x) for x in xs]
    loss = sum([(yp-yg)**2 for yp,yg in zip(ypred,ys)])
    loss.backward()
    for p in n.parameters():
        p.data -= 0.1*p.grad
        p.grad = 0
    if not i % 100:
        print(loss)

print(0 if n([1,1]).data<0.5 else 1)