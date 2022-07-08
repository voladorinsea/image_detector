# there are 5 kinds of layers used in yolo
# [convolutional]
 batch_normalize=1  
 filters=64  
 size=3  
 stride=1  
 pad=1  
 activation=leaky

# [shortcut]
 from=-3  
 activation=linear  
 [p.s. A shortcut layer is a skip connection, like the one used in ResNet.]
 [The from parameter is -3, which means the output of the shortcut layer is obtained by adding feature maps from the previous and the 3rd layer backwards from the shortcut layer.]

# [Upsample]
stride=2
[p.s. Using bilinear method to upsample]
# [Route]
layers = -4
or
layers = -1, 61
[When layers attribute has only one value, it outputs the feature maps of the layer indexed by the value. In our example, it is -4, so the layer will output feature map from the 4th layer backwards from the Route layer.]
[When layers has two values, it returns the concatenated feature maps of the layers indexed by it's values. In our example it is -1, 61, and the layer will output feature maps from the previous layer (-1) and the 61st layer, concatenated along the depth dimension.]
# [yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .5
truth_thresh = 1
random=1
[mask is uesd to select the dimension of anchors]
# [net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=16
width= 320
height = 320
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1
[it only describes information about the network input and training parameters. It isn't used in the forward pass of YOLO. ]