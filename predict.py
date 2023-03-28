# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from model import Model
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

def get_img_by_url(url):
    response = requests.get(url).content
    nparr = np.frombuffer(response, np.uint8)
    # convert to image array
    img = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)
    return img

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
        characters: str = Input(description="secondary characters", default="sheeps"),
        pose: str = Input(description="sitting, standing, fighting, sleeping, standing_profile, standing_back, sitting_profile, screaming, running_right, running_left, reading, going_right, going_left, crying", default="sitting"),
        reference: str = Input(description="reference image", default=""),
    ) -> Path:
        """Run a single prediction on the model"""
        characters_map = {
            'MAKAL_3371': self.boyModel,
            'taySwift': self.womanModel,
            'KEAR_3371': self.manModel,
            'MKENNA_3371': self.girlModel
        }

        emotions_map = {
            'scream': 'screaming face, open mouth, loud shouting, feeling terrified or angry',
            'normal': 'neutral expression, calm demeanor, composed body language, feeling at ease, straight face',
            'fun': 'happy attitude, fun mood, happy laughing, light smile, fun and kind eyes',
            'joy': 'fun and kind eyes, joy, joyful expression, smiling face, laughing out loud, feeling ecstatic',
            'anger': 'anger, angry expression, frowning, gritting teeth, clenching fists, shaking with rage, angry eyes, wrinkled',
            'weeping': 'lowered lip corners, sad, weeping expression, crying profusely, wiping tears, hunched over, feeling heartbroken',
            'sad': 'sad, sad face, lowered lip corners, no smile, sad expression, tears, droopy and sad eyes, feeling despondent, upset',
            'surprise': 'surprised, ajar mouth, dramatic, widest eyes, happy, surprise, feeling startled, shocked',
            'fear': 'fear, fearful, curve, shock, shocked, sad, scared, wide opened eyes, widest eyes, opened mouth, scared',
        }

        styles_map = {
            'painting': 'Masterpiece, cinematic lighting, photorealistic, realistic, extremely detailed, artgerm, greg rutkowski, alphonse mucha',
            'anime': 'modern anime style art detailed shinkai makoto vibrant Studio animation hideaki anno Sakimichan Stanley Artgerm Lau Rossdraws James Jean Marc Simonetti elegant digital painting artstation'
        }

        poses_map = { 
            'medium': {
                'standing': './poses/standing_medium.png',
            },
            'full': {
                'standing': './poses/standing_full.png',
            },
            'close': {
                'standing': './poses/close_sitting_standing.png',
            }
        }         
        
        if isWarmup=="true":
            return
        
        prompt = f"{character} as {description}, {styles_map[style]}, (((white shirt, white pants))), {emotions_map[emotion]}, on background {environment}, {characters}"
        pose_input = cv2.imread(poses_map[shot][pose], 0) if reference.strip() == "" else get_img_by_url(reference)
        n_prompt = "Ugly, lowres, duplicate, morbid, mutilated, out of frame, extra fingers, extra limbs, extra legs, extra heads, extra arms, extra breasts, extra nipples, extra head, extra digit, poorly drawn hands, poorly drawn face, mutation, mutated hands, bad anatomy, long neck, signature, watermark, username, blurry, artist name, deformed, distorted fingers, distorted limbs, distorted legs, distorted heads, distorted arms, distorted breasts, distorted nipples, distorted head, distorted digit"
        a_prompt = ""
        num_samples = 1
        image_resolution = 768
        detect_resolution = 768
        ddim_steps = 20
        scale = 12
        eta = 0
        outputs = characters_map[character].process_depth(pose_input, prompt, a_prompt, n_prompt,
                     num_samples, image_resolution, detect_resolution,
                     ddim_steps, scale, seed, eta)
        boy_image = cv2.cvtColor(outputs[1], cv2.COLOR_BGR2RGB)
        cv2.imwrite("output.png", boy_image)
        return Path("output.png")
