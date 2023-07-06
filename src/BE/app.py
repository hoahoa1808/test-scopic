import uvicorn
from mangum import Mangum
from fastapi.responses import JSONResponse
from fastapi import File, UploadFile, FastAPI, Request
import os
import time
from typing import List
import random
from PIL import Image
import io
import subprocess


app=FastAPI()

###############################################################################
#   Routers configuration                                                     #
###############################################################################

@app.get("/")
async def root(request: Request):
    print(request)
    return {"message":  f"experiment API in: {str(request.url)}docs",
            "Machine configs-CPU": os.cpu_count() }

@app.post("/")
async def root(request: Request):
    print(request.json())
    return {"message":  f"experiment API in: {str(request.url)}docs",
            "Machine configs-CPU": os.cpu_count() }


@app.post("/prompt", summary="extract layout of pdf file by detection model")
async def post_prompt_2_image(prompt:str, num_sample:int, negative_prompt:str="", weight:int=256, height:int=256):
    try:
        images=[]

        res = JSONResponse({
            "status": "success",
            "data": images,
        })
        return res
    except Exception as e:
        return JSONResponse({
            "status": "false",
            "error": e
        })

###############################################################################
#   Handler for AWS Lambda                                                    #
###############################################################################

handler = Mangum(app)

###############################################################################
#   Run the self contained application                                        #
###############################################################################

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=9000)