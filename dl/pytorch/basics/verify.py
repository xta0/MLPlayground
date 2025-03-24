from helper import verify

print(verify('./tests/nhwc_y.txt', './tests/mps_y1_conv.txt'))
print(verify('./tests/nchw_y.txt', '../cnn/cifar10/mps/conv1_nchw.txt'))
print(verify('./tests/mps_y1_conv.txt', '../cnn/cifar10/mps/conv1_nhwc.txt'))
print(verify('./tests/mps_y1_relu.txt', '../cnn/cifar10/mps/relu1_nhwc.txt'))
print(verify('./tests/mps_y1_mp.txt', '../cnn/cifar10/mps/pool1_nhwc.txt'))

print(verify('./tests/mps_fc1.txt', '../cnn/cifar10/mps/fc1.txt'))
print(verify('./tests/mps_fc2.txt', '../cnn/cifar10/mps/fc2.txt'))
print(verify('./tests/mps_softmax.txt', '../cnn/cifar10/mps/softmax.txt'))
