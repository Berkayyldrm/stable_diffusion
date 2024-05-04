import io
from PIL import Image
from PIL import ImageColor
from fastapi import File, UploadFile, Form, APIRouter
from fastapi.responses import StreamingResponse

from app.models.stable_diffusion import StableDiffusionFactory
router = APIRouter()

@router.post("/create_image")
async def process_image(
    prompt: str = Form(...),
    color_image_hex: str = Form(...), 
    ad_image: UploadFile = File(...)):

    model = StableDiffusionFactory.get_model()
    contents = ad_image.file.read()
    image_stream = io.BytesIO(contents)
    image = Image.open(image_stream).convert("RGB")
    image_stream.seek(0)
    image = image.resize((768, 512))
    
    prompt_color = f"colored with RGB in {ImageColor.getcolor(color=color_image_hex, mode='RGB')}"
    new_image = await model.transform_image(image, prompt+ " " +prompt_color)
    img_byte_arr = model.get_byte_image(new_image)
    return StreamingResponse(img_byte_arr, media_type="image/png")

@router.post("/create_advert")
async def process_advert(
    ad_image: UploadFile = File(...),
    prompt: str = Form(...),
    color_image_hex: str = Form(...), 
    logo: UploadFile = File(...),
    color_text_hex: str = Form(...),
    punchline_text: str = Form(...),
    button_text: str = Form(...)):

    model = StableDiffusionFactory.get_model()
    contents = ad_image.file.read()
    image_stream = io.BytesIO(contents)
    image = Image.open(image_stream).convert("RGB")
    image_stream.seek(0)
    image = image.resize((768, 512))

    contents = logo.file.read()
    logo_stream = io.BytesIO(contents)
    logo = Image.open(logo_stream).convert("RGB")
    logo_stream.seek(0)

    prompt_color = f"colored with RGB in {ImageColor.getcolor(color=color_image_hex, mode='RGB')}"
    processed_image = model.transform_image(image, prompt+ " " +prompt_color)
    new_image = model.create_ad_template(processed_image, logo, color_text_hex, punchline_text, button_text)
    img_byte_arr = model.get_byte_image(new_image)

    return StreamingResponse(img_byte_arr, media_type="image/png")