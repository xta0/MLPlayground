from __future__ import print_function
import numpy as np
import torch

#Construct a randomly initialized matrix:
x = torch.rand(5, 3)
print(x)
print(x.shape) #[5,3] 

#Construct a matrix filled zeros and of dtype long:
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

#Construct a tensor directly from data:
x = torch.tensor([5.5, 3])
print(x.shape)
print(x.size())

#ops
x = torch.ones(5,3)
y = torch.rand(5,3)
print(x+y)
y.add_(x) #in-place operation

#Resizing: If you want to resize/reshape tensor, you can use torch.view:
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8) ## the size -1 is inferred from other dimensions
print(x.size(),y.size(),z.size())

#Storage
points = torch.tensor([[1.0,4.0], [2.0,3.0], [5.0,4.0]])
storage = points.storage()
print(storage) #storage is a one dimentional array [1.0, 4.0, 2.0, 3.0, 5.0, 4.0]
storage[0] = 10.0
storage[0] = 1.0
print(points) #change storage will change the coresponding tensor

#size
print("points.size: ", points.size()) #size is tensor's shape ([3,2]) 

#shape
print("points.shape: ", points.shape) #([3,2]) 

#storage offset
point1 = points[1]
print("point1: ", point1) #[2.0, 3.0]
print("point1.storage_offset() = ", point1.storage_offset()) #2
#storage offset the is index in the storage that corresponds to the first element in the tensor 
#[1.0, 4.0, 2.0, 3.0, 5.0, 4.0]
#           |point1's index is 2
#                 |point1_y's index is 3
point1_y = point1[1] #tensor(3.)
print("point1_y: ", point1_y) #3
print("point1_y.storage_offset: ", point1_y.storage_offset())

#stride
#stride is the number of elements in the storage that needs to be skipped to obtain the next element along each dimension
stride = points.stride()
print("points' stride: ", stride) #(2,1) skip two elements on X-axis, skip one element on Y-axis
#---(2)----
#[1.0, 4.0, | 
# 2.0, 3.0,(1)
# 5.0, 4.0] |
#the main purpose of stride is to covert a 2d point to the index in storage
#say accessing an element (i,j) in a 2d tensor results in accessing the 
#index = offset + stride[0]*i + stride[1]*j
#offset will usually be zero

#sub tensor
#point1 is a sub tensor of points
#chaning point1 will affect points
point1[0] = 100
print(points)
point1[0] = 1.0
#to avoid the side effect, we can use clone
point0 = points[0].clone()
point0[0] = 100
print(points)

#transform
points_T = points.t()
print("points' transform: ", points_T)
#transform wont cause any memory reallocation, just change the shape and stride
#we can verify the storage
print(id(points.storage()) == id(points_T.storage()))
print("points_T's stride: ", points_T.stride()) #(1,2)
#after transform tensor's storage becomes incontiguous
print("points_T's contiguous: ", points_T.is_contiguous()) #False 
#you can make it contiguous again, by calling contiguous()
points_T_con = points_T.contiguous()
print("points_T_con's stride: ", points_T_con.stride()) #(3,1)
print("points_T_con's storage: ", points_T_con.storage()) #[1.0,1.0,5.0,4.0,3.0,4.0]

#dtype
double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)

#numpy
#torch tensor and numpy array shares the same memory storage if the backend is CPU
points = torch.ones(3, 4)
points_np = points.numpy()
print("numpy array: ", points_np)

#serializing tensors
#save tensor to pickle file
torch.save(points, "./points.pk")
points = torch.load("./points.pk")

#view
b = points.view(2,6)
print(b)