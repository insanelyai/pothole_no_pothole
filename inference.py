import torch
from torchvision import transforms, models
from PIL import Image
import sys

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("pothole_model.pth", map_location=device))
model.to(device)
model.eval()

# Transform (must match validation transform)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

classes = ["no_pothole", "pothole"]


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return classes[predicted.item()], confidence.item()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py image.jpg")
    else:
        image_path = sys.argv[1]
        label, confidence = predict(image_path)
        print(f"\nPrediction: {label}")
        print(f"Confidence: {confidence:.4f}")
