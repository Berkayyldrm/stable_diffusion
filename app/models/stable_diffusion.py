import io
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image, ImageDraw, ImageFont

class StableDiffusion():
    def __init__(self):
        self.model = None
        self.device = "cuda"
        self.model_id_or_path = "ml_model"
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_id_or_path, torch_dtype=torch.float16).to(self.device)
    
    def transform_image(self, ad_image, prompt):
        new_image = self.pipe(prompt=prompt, image=ad_image, strength=0.75, guidance_scale=7.5).images[0]
        new_image.save("./test_images/test.png")

        return new_image
    
    def create_ad_template(self, ad_image, logo, color_hex, punchline_text, button_text):
        template_width, template_height = 800, 600
        logo_size, ad_image_size = (100, 100), (256, 256)
        logo = logo.resize(logo_size)
        ad_image = ad_image.resize(ad_image_size)
        template = Image.new('RGB', (template_width, template_height), 'white')
        draw = ImageDraw.Draw(template)

        logo_position = ((template_width - logo_size[0]) // 2, 20)  # Logo merkezleniyor
        template.paste(logo, logo_position, logo if logo.mode == 'RGBA' else None)
        ad_image = ad_image.resize(ad_image_size)
        ad_image_position = ((template_width - ad_image_size[0]) // 2, logo_position[1] + logo_size[1] + 20)
        template.paste(ad_image, ad_image_position)

        font = ImageFont.load_default()
        punchline_size = draw.textsize(punchline_text, font=font)
        punchline_position = ((template_width - punchline_size[0]) // 2, ad_image_position[1] + ad_image_size[1] + 20)

        button_size = (200, 50)
        button_position = ((template_width - button_size[0]) // 2, punchline_position[1] + punchline_size[1] + 10)
        
        # Metni ve butonu çizme
        draw.text(punchline_position, punchline_text, fill=color_hex, font=font)
        draw.rectangle([button_position[0], button_position[1], button_position[0] + button_size[0], button_position[1] + button_size[1]], outline=color_hex, fill=color_hex)
        button_text_size = draw.textsize(button_text, font=font)
        button_text_position = (button_position[0] + (button_size[0] - button_text_size[0]) // 2, button_position[1] + (button_size[1] - button_text_size[1]) // 2)
        draw.text(button_text_position, button_text, fill='white', font=font)

        template.save("./test_images/ad_template.png")
        return template
    
    def get_byte_image(self, image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return img_byte_arr
    

class StableDiffusionFactory:
    _model = None

    @staticmethod
    def get_model():
        if StableDiffusionFactory._model is None:
            StableDiffusionFactory._model = StableDiffusion()
        return StableDiffusionFactory._model
