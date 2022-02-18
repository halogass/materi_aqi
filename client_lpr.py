from pprint import pprint
import requests
import base64
import numpy as np
import cv2
from matplotlib import pyplot as plt

imPathCoba = ['./media/mobil.jpg']

intloopRequest = 0

def showHasil(imgHasil):
    cv2.imshow('Inference', imgHasil)
    cv2.waitKey() & 0xFF == ord('q')

while intloopRequest <= 0:
    for fileItem in imPathCoba:
        files = {'file': open(fileItem, 'rb')}
        response = requests.post('http://api.arsa.technology:5402/v0/lpr?imOut=true&superRes=false', files=files)
        inputRes = response.json()
        inputImgRaw = inputRes['result_img']
        nparr = np.frombuffer(base64.b64decode(inputImgRaw), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        plt.imshow(img)
        plt.show()
        pprint(inputRes)
    intloopRequest += 1


