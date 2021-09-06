from fastapi import FastAPI, Form,UploadFile
import os
import time
import stat
import uuid
from pydantic import BaseModel
import numpy as np
import cv2
from fastapi.staticfiles import StaticFiles #new
from deepface import DeepFace
app = FastAPI(title="TelecomPOC",version="1.0.0")
app.mount("/images", StaticFiles(directory="app/images"),name="images")

face_class= cv2.CascadeClassifier(os.getcwd()+"/app/haarcascade_frontalface_default.xml")
def face_extractor(photo):
    gphoto=cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    detected=face_class.detectMultiScale(gphoto)
    if len(detected)==0:
        return None
    else:
        (x,y,w,h)=detected[0]
        cphoto=photo[y:y+h,x:x+w]
        return cphoto
@app.get("/")
async def root():
    return {"message": "server is up"}

@app.post("/addToTrainQueue")
async def addToTrainQueue( photo: UploadFile = Form(...),mobileNumber: str=Form(...)):
    fname=str(uuid.uuid4()).replace(" ", "-")
    if not os.path.exists(os.getcwd()+"/app/images/"+mobileNumber):
        os.makedirs(os.getcwd()+"/app/images/"+mobileNumber)
    content = await photo.read()
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cphoto=face_extractor(img)
    face=cv2.resize(cphoto, (300,300),interpolation=cv2.INTER_AREA)
    face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.getcwd()+"/app/images/"+mobileNumber+"/"+fname+".jpg", face)
    return {"status":"success","filename": mobileNumber}

@app.post("/searchImage")
async def searchImage( photo: UploadFile = Form(...)):
    fname=str(uuid.uuid4()).replace(" ", "-")
    content = await photo.read()
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    face=cv2.resize(img, (500,500),interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.getcwd()+"/app/verification/"+fname+".jpg", face)
    df=DeepFace.find(img_path = os.getcwd()+"/app/verification/"+fname+".jpg", db_path = os.getcwd()+"/app/images/",enforce_detection =False)
    os.remove(os.getcwd()+"/app/verification/"+fname+".jpg")
    return {"status":"success","filename": df}

@app.get("/retrain")
async def retrain():
    os.remove(os.path.join(os.getcwd()+"/app/images", "representations_vgg_face.pkl"))
    df=DeepFace.find(img_path = os.getcwd()+"/app/images/8970412007/dfbfa83b-82e6-45af-992e-1392de505273.jpg", db_path = os.getcwd()+"/app/images/",enforce_detection =False)
    return {"status":"success","message":"training triggered, you may not see results for couple of mins"}

@app.get("/getLastSuccessfulTimeStamp")
async def getLastSuccessfulTimeStamp():
    if os.path.exists(os.path.join(os.getcwd()+"/app/images", "representations_vgg_face.pkl")):
        fileStatsObj = os.stat(os.path.join(os.getcwd()+"/app/images", "representations_vgg_face.pkl"))
        modificationTime = time.ctime ( fileStatsObj [ stat.ST_MTIME ] )
        return {"status":"success","message":modificationTime}
    else:
        return {"status":"failed","message":"never trained"}