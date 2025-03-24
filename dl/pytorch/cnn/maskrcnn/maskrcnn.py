import torch
import torchvision
import cv2
import argparse
from PIL import Image
from utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('-i',
                    '--input',
                    required=True,
                    help='path to the input data')
parser.add_argument('-t',
                    '--threshold',
                    default=0.965,
                    type=float,
                    help='score threshold for discarding detection')
args = vars(parser.parse_args())

# initialize the model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, 
                                                           num_classes=91)
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()

# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

image_path = args['input']
image = Image.open(image_path).convert('RGB')
# keep a copy of the original image for OpenCV functions and applying masks
orig_image = image.copy()
# transform the image
image = transform(image)
# add a batch dimension
image = image.unsqueeze(0).to(device)
print("image:", image.shape)
# with torch.no_grad():
# forward pass of the image through the modle
outputs = model(image)
masks, boxes, labels = get_outputs(outputs, args['threshold'])
print("masks: ", masks.shape)
result = draw_segmentation_map(orig_image, masks, boxes, labels)
# visualize the image
# cv2.imshow('Segmented image', result)
# cv2.waitKey(0)
# set the save path
save_path = f"./outputs/{args['input'].split('/')[-1].split('.')[0]}.jpg"
cv2.imwrite(save_path, result)

# trace model
# scripted_model = torch.jit.trace(model, image)
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "./scripted_maskrcnn.pt")
scripted_model._save_for_lite_interpreter("./maskrcnn_mobile.pt")
print(scripted_model)
ops = torch.jit.export_opnames(scripted_model)
print(ops)
