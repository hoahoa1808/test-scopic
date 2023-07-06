import uvicorn
from mangum import Mangum
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request
import os
import io
import datetime
import cloudinary
import cloudinary.uploader
import cloudinary.api
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from main import infer

cloudinary.config( 
  cloud_name = 'int3306uet', 
  api_key= '748356266443834', 
  api_secret= 'B9lB62U3x-Kyqxpfzp5EGGfoHFU' )
app=FastAPI()
handler = Mangum(app)

#CORS
origins = [
    "http://localhost",
    "http://localhost:8080",
    "null"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
###############################################################################
#   Routers configuration                                                     #
###############################################################################

@app.get("/")
async def root(request: Request):
    print(request)
    return {"message":  f"experiment API in: {str(request.url)}docs",
            "Machine configs-CPU": os.cpu_count() }

def up_img_to_cloud(img, img_id):
    url = '/home/'+ str(img_id)
    r = cloudinary.uploader.upload(img, folder=url)
    return r['url']

@app.post("/prompt", summary="extract layout of pdf file by detection model")
async def post_prompt_2_image(request:Request):
# async def post_prompt_2_image(prompt:str, num_sample:int, negative_prompt:str="", weight:int=256, height:int=256):
    request = await request.body()
    decoded_request = request.decode("utf-8") 
    values = {i.split("=")[0]:i.split("=")[-1]for i in decoded_request.split("&")}
    for k in values:
        rollback = values[k]
        try: values[k] = int(values[k])
        except : values[k] = rollback

    try:
        print(values)
        # Generate images dynamically
        # image = Image.open("/home/k64t/T2I/ImageReward/assets/images/2.webp")
        # images=[image]
        images = infer(prompt=values['prompt'].replace("+", " "),
                negative_prompt=values['negative_prompt'], 
                num_samples=values['num_samples'], W=values['width'],H=values['height'])
        print(images)

        # Convert images to base64-encoded strings
        base64_images = []
        for image in images:
            # Convert PIL Image to bytes
            with io.BytesIO() as buffer:
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                folder = "{}".format(datetime.datetime.now().strftime("%Y-%M-%D"))
                url = up_img_to_cloud(image_bytes, folder)
                
            # Encode image bytes as base64
            base64_images.append(url)


        return JSONResponse({
            "status": "success",
            "images": base64_images,
        })
    except Exception as e:
        return JSONResponse({
            "status": "false",
            "error": e
        })







if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8080)