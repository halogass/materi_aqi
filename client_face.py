from pprint import pprint
import requests
import base64
import numpy as np
import cv2
from matplotlib import pyplot as plt

imPathCoba = ['./media/AQI_test.jpeg', './media/AQI_knownface.jpeg', './media/id.jpeg', './media/knownFace.jpeg', './media/faces.jpg']
host = 'http://api.arsa.technology:3500'
intloopRequest = 0
def testVerify():
    files = {
        'photoFile1': open(imPathCoba[0], 'rb'),
        'photoFile2': open(imPathCoba[3], 'rb')}
    response = requests.post(host + '/v0/verify', files=files)
    inputRes = response.json()
    pprint(inputRes)

def testIdVerify():
    files = {
        'idPhoto': open(imPathCoba[2], 'rb'),
        'subjectPhoto': open(imPathCoba[0], 'rb')}
    response = requests.post(host + '/v0/id_verify', files=files)
    inputRes = response.json()
    pprint(inputRes)

def testFaceEmbed():
    files = {
        'photoFile': open(imPathCoba[4], 'rb')
    }
    imgOut = 'true'
    superRes = 'false'
    response = requests.post((host+'/v0/face_embed?'+'imgOut='+imgOut+'&superRes='+superRes) , files=files)
    inputRes = response.json()
    result = inputRes['result']
    for face in result:
        pprint(face)
        faceMbed = np.array(result[str(face)]['embeddings'])
        bbox = result[str(face)]['bbox']
        pprint(bbox)
        pprint(faceMbed)
    if imgOut == 'true':
        inputImgRaw = inputRes['result_img']
        nparr = np.frombuffer(base64.b64decode(inputImgRaw), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        plt.imshow(img)
        plt.show()

#testVerify()
#testIdVerify()
testFaceEmbed()

