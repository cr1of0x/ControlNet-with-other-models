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
        self.womanModel = Model('sd-dreambooth-library/taylorswift', 'Taylor_Berry.ckpt')
        self.girlModel = Model('criofox/MKENNA_3771', 'MKENNA_3371.ckpt')
        self.boyModel = Model('criofox/MAKAL_3371', 'MAKAL_3371.ckpt')
        self.manModel = Model('criofox/KEAR_3371', 'KEAR_3371.ckpt')

    def predict(
        self,
        style: str = Input(description="Style of model", default='painting'),
        character: str = Input(description="Character instance", default='MKENNA_3371'),
        description: str = Input(description='Character description(shepard boy)', default="shepard girl"),
        seed: str = Input(description="seed", default="1337"),
        emotion: str = Input(description="fun or normal", default="normal"),
        shot: str = Input(description="close, full, medium", default="medium"),
        environment: str = Input(description="environment", default="hillside, green grass, sunny day"),
        isWarmup: str = Input(description="need be setted false", default="false"),
        characters: str = Input(description="secondary characters", default="sheeps")
    ) -> Path:
        """Run a single prediction on the model"""
        characters_map = {
            'MAKAL_3371': self.boyModel,
            'taySwift': self.womanModel,
            'KEAR_3371': self.manModel,
            'MKENNA_3371': self.girlModel
        }

        emotions_map = {
            'fun': 'happy attitude, fun mood, happy laughing',
            'normal': 'normal attitude, normal mood, straight face'
        }

        styles_map = {
            'painting': 'Masterpiece, cinematic lighting, photorealistic, realistic, extremely detailed, artgerm, greg rutkowski, alphonse mucha',
            'anime': 'modern anime style art detailed shinkai makoto vibrant Studio animation hideaki anno Sakimichan Stanley Artgerm Lau Rossdraws James Jean Marc Simonetti elegant digital painting artstation'
        }

        poses_map = { 
            'medium': {
                'sitting': './poses/sitting_medium.png', 
                'standing': './poses/standing_medium.png'
            },
            'full': {
                'sitting': './poses/sitting_full.png', 
                'standing': './poses/standing_full.png'
            },
            'close': {
                'sitting': './poses/close_sitting_standing.png', 
                'standing': './poses/close_sitting_standing.png'
            }
        }         
        
        if isWarmup=="true":
            return
        
        prompt = f"{character} as {description}, {styles_map[style]}, (((white shirt, white pants))), {emotions_map[emotion]}, on background {environment}, {characters}"
        pose = cv2.imread(poses_map[shot][pose], 0)
        n_prompt = "Ugly, lowres, duplicate, morbid, mutilated, out of frame, extra fingers, extra limbs, extra legs, extra heads, extra arms, extra breasts, extra nipples, extra head, extra digit, poorly drawn hands, poorly drawn face, mutation, mutated hands, bad anatomy, long neck, signature, watermark, username, blurry, artist name, deformed, distorted fingers, distorted limbs, distorted legs, distorted heads, distorted arms, distorted breasts, distorted nipples, distorted head, distorted digit"
        num_samples = 1
        image_resolution = 768
        detect_resolution = 768
        ddim_steps = 20
        scale = 12
        eta = 0
        outputs = characters_map[character].process_pose(pose, prompt, a_prompt, n_prompt,
                     num_samples, image_resolution, detect_resolution,
                     ddim_steps, scale, seed, eta)
        boy_image = cv2.cvtColor(outputs[1], cv2.COLOR_BGR2RGB)
        cv2.imwrite("output.png", boy_image)
        return Path("output.png")
