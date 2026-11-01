import torch
import os

class MyScriptModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def myfn(self, x):
        if x.dim() == 1:
            return torch.ones(10)
        else:
            return torch.zeros(10)

    def forward(self, x):
        return self.myfn(x)


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(10, 32, bias=False),)
        self.scriptModule = MyScriptModule()

    def forward(self, x):
        x = self.scriptModule(x)
        return self.net(x)


def run(x):
    model = torch._inductor.aoti_load_package(os.path.join(os.getcwd(), "model.pt2"))
    y = model(x)
    print(y)



def main():
    x1 = torch.randn(3, 10)
    x2 = torch.randn(10)
    print(x1.dim())
    print(x2.dim())
    model = MyModule()
    y1 = model(x1)
    y2 = model(x2)
    print(y1)
    print(y2)
    # this traces the model and produces a static graph 
    exported_model = torch.export.export(model, args=(x1,),)
    print(exported_model)
    torch._inductor.aoti_compile_and_package(exported_model, package_path=os.path.join(os.getcwd(), "model.pt2"))

    run(x2)

if __name__ == "__main__":
    main()
