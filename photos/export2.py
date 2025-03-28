import torch
import coremltools as ct
from torchvision import models
import torch.nn as nn
import numpy as np

# 1. Load your trained model checkpoint
checkpoint = torch.load("face_classifier_resnet18.pt", map_location="cpu")
num_classes = checkpoint["num_classes"]

# 2. Rebuild the exact same model architecture used during training
#    (including final nn.Softmax(dim=1))
model = models.mobilenet_v3_small(pretrained=False)
num_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes),
    nn.Softmax(dim=1)  # <-- Final layer outputs probabilities
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 3. Trace the model with a sample input
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# 4. Define the input type for Core ML
tensor_input = ct.TensorType(
    name="input", 
    shape=(1, 3, 224, 224),
    dtype=np.float32
)

# 5. Convert to a Core ML model without any classifier configuration
#    => outputs a single MultiArray of probabilities
mlmodel = ct.convert(
    traced_model,
    inputs=[tensor_input]
)

print(mlmodel)

# 6. Add metadata and save
mlmodel.input_description["input"] = "Input tensor (1, 3, 224, 224)."
# The actual output name might be something like "output" or "classifier.1".
spec = mlmodel.get_spec()
output_name = spec.description.output[0].name  # The first (and only) output

mlmodel.output_description[output_name] = "Softmax probabilities for each class (index-based)."
mlmodel.author = "Tao"
mlmodel.license = "MIT"
mlmodel.short_description = "MobileNetV3 face classifier that outputs probabilities directly."
mlmodel.save("FaceClassifier.mlpackage")

print("Core ML model exported to FaceClassifier.mlpackage")
