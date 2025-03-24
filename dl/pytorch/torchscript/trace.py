import torch
from typing import Dict


# @torch.jit.script
# def myfn(x, mask: Dict[int, int]):
#     if x.dim() == 1:
#         return torch.ones(10)
#     else:
#         return torch.zeros(10)


# inp1 = torch.randn(1)
# inp2 = torch.randn(())
# mask: Dict[int, int] = {}
# mask[0] = 1
# mask[1] = 2
# print(myfn(inp1, mask))
# print(myfn(inp2, mask))
# traced_fn = torch.jit.trace(myfn, (inp1, mask))
# # traced_fn = torch.jit.trace(myfn, inp1)
# print(traced_fn.graph)
# print(traced_fn.code)
# print(traced_fn(inp1))
# print(traced_fn(inp2))


class MyScriptModule(torch.nn.Module):
    def __init__(self):
        super.__init__()

    def myfn(self, x):
        if x.dim() == 1:
            return torch.ones(10)
        else:
            return torch.zeros(10)

    def foward(self, x):
        return self.my(x)


class MyModule(torch.nn.Module):
    def __init__(self):
        super.__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(10, 256))
        self.scriptModule = MyScriptModule()

    def foward(self, x):
        x = self.scriptModule(x)
        return self.net(x)


# x1 = torch.randn(1)
# x2 = torch.randn(())

# model = MyModule()
# y1 = model(x1)
# y2 = model(x2)


class MyModule2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(512, 256),
                                       torch.nn.PReLU(),
                                       torch.nn.Linear(256, 128),
                                       torch.nn.LogSoftmax(dim=1))

    def forward(self, x):
        return self.net(x)


model2 = MyModule2()
x = torch.randn([1, 512])
y = model2(x)
print(y)
t1 = torch.jit.script(model2)
print(type(t1))
print(t1.graph)
print(t1.code)
t2 = torch.jit.trace(model2, x)
print(type(t2))
print(t2.graph)
print(t2.code)
