import torch
import cv2
from PIL import Image
import numpy as np
from cog import BasePredictor, Input, Path

import utils
from midas.model_loader import load_model


MODEL_CACHE = "weights"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        models = [
            "dpt_beit_large_512",
            "dpt_swin2_large_384",
            "dpt_swin2_tiny_256",
            "dpt_levit_224",
        ]
        self.models = {
            model: load_model(
                self.device, model_path=f"weights/{model}.pt", model_type=model
            )
            for model in models
        }

    def predict(
        self,
        model_type: str = Input(
            default="dpt_beit_large_512",
            choices=[
                "dpt_beit_large_512",
                "dpt_swin2_large_384",
                "dpt_swin2_tiny_256",
                "dpt_levit_224",
            ],
        ),
        image: Path = Input(description="Input image"),
    ) -> Path:
        """Run a single prediction on the model"""
        model, transform, net_w, net_h = self.models[model_type]
        model.to(self.device)
        model.eval()

        original_image_rgb = utils.read_image(str(image))
        image = transform({"image": original_image_rgb})["image"]
        target_size = original_image_rgb.shape[1::-1]

        with torch.no_grad():
            sample = torch.from_numpy(image).to(self.device).unsqueeze(0)

            height, width = sample.shape[2:]
            prediction = model.forward(sample)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output_path = "/tmp/out.png"
        output = prediction.cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        img = Image.fromarray(formatted)
        img.save(output_path)

        return Path(output_path)
