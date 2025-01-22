import torch
a = torch.cuda.get_device_name(0)  # 返回GPU名字
print("a is ", a)
b = torch.cuda.get_device_name(1)
print("b is ", b)
c = torch.cuda.get_device_name(2)
print("c is ", c)
d = torch.cuda.get_device_name(3)
print("d is ", d)
