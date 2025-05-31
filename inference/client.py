import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import tritonclient.http as httpclient

class_labels = ["Healthy", "Osteopenia", "Osteoporosis"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_image(img_path):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.numpy()
    return img


def run_onnx(img, http_client):
    classification_input = httpclient.InferInput("x", img.shape, datatype="FP32")
    classification_input.set_data_from_numpy(img, binary_data=True)

    classification_response = http_client.infer(
        model_name="knee_classification_onnx", inputs=[classification_input]
    )

    scores = classification_response.as_numpy("linear_2")
    return scores


if __name__ == "__main__":
    client = httpclient.InferenceServerClient(url="localhost:8000")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--impath", help="Path to image to classify", required=True
    )
    args = parser.parse_args()

    impath = args.impath
    preprocessed_img = read_image(impath)

    output = run_onnx(preprocessed_img, client)
    preds = np.argmax(output)
    print(class_labels[preds])
