# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from model import Model
import cv2
import numpy as np
import matplotlib.pyplot as plt




class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.manModel = Model('criofox/KEAR_3371', 'KEAR_3371.ckpt')

    def predict(
        self,
        prompt: str = Input(description="Prompt for the model"),
    ) -> Path:
        """Run a single prediction on the model"""
        img = cv2.imread('./poses/sitting.png', 0)
        m_prompt = "KEAR_3371 as sheppard boy, modern anime style art detailed shinkai makoto vibrant Studio animation hideaki anno Sakimichan Stanley Artgerm Lau Rossdraws James Jean Marc Simonetti elegant highly detailed digital painting artstation (((dirty poor shirt, dirty pants))), sad attitude, shocked boy, crying boy, on background green grass hillside"
        girlPrompt = "taySwift as sheppard girl, Masterpiece, cinematic lighting, photorealistic, realistic, extremely detailed, (((dirty poor shirt, dirty pants))), sad attitude, shocked girl, crying girl, artgerm, greg rutkowski, alphonse mucha, on background green grass hillside"
        a_prompt = ""
        n_prompt = "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"
        num_samples = 1
        image_resolution = 768
        detect_resolution = 768
        ddim_steps = 20
        scale = 12
        seed = 1337
        eta = 0
        outputs = manModel.process_pose(img, prompt, a_prompt, n_prompt,
                     num_samples, image_resolution, detect_resolution,
                     ddim_steps, scale, seed, eta)
        outputs = [Image.fromarray(output) for output in outputs]
        # save outputs to file
        outputs = [output.save(f"tmp/output_{i}.png") for i, output in enumerate(outputs)]
        # return paths to output files
        return [Path(f"tmp/output_{i}.png") for i in range(len(outputs))]
