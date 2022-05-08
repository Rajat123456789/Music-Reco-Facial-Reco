from admin_authentication import data_upload, admin_signup1, admin_login1, user_extract, retrieval, userExists, user_extract_language
from flask import Flask,flash,render_template,request,session, url_for, redirect, session,url_for

#from flask_mysqldb import MySQL
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#from sklearn import datasets
import pickle
import pymysql
import numpy as np
import math
import pickle
import collections
import cv2
import time
import numpy as np
import pandas as pd
import os


import pandas as pd
import cv2
import numpy as np

import numpy as np

import pandas as pd
import re
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf


from sklearn import preprocessing 
  


import spotipy
# cid = 'eb9d53df11484a72b61c48259ffd7c8b'
# secret = '33f5f097a1404c8082bf1ce05b9c23bf'
# from spotipy.oauth2 import SpotifyOAuth
# spotify = spotipy.Spotify(auth_manager = SpotifyOAuth(client_id = cid,
#                                                       client_secret =secret,
#                                                       redirect_uri = 'http://localhost:5000/callback'))
cid="0f0b1633f7f74a0c937e46d42de6497c"
secret="4fd5e59894cf459686ff971fd0731b6a"
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials

#spotify = spotipy.Spotify(auth_manager = SpotifyOAuth(client_id = cid,
                                                      #client_secret =secret,
                                                      #redirect_uri = 'http://localhost:5000/callback'))
auth_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
spotify = spotipy.Spotify(auth_manager=auth_manager)
def valid_token(resp):
    return resp is not None and not 'error' in resp


def make_search_own(search_type, name):
    print("hihi")
    data = spotify.search(name,limit=20,type="track")   
    api_url = data['tracks']['items']
    items = data['tracks']['items']
    #print(items)
        

    # #    return render_template('prediction.html',user=lan)
    # else:
    #     return render_template('login.html')



    p=[]
    for i in range(len(items)):
        b=items[i]['id']
        p.append(b)
    import random
    items = random.sample(items, len(items))
    print("hiohiohihihi")
    maxindex1="no emotion"

    return render_template('search1.html',
                           name=name,
                           results=items,
                           api_url=api_url, 
                           search_type=search_type)



def make_search(search_type, name):
    global maxindex1
    print("hihi")
    data = spotify.search(name,limit=20,type="track")   
    api_url = data['tracks']['items']
    items = data['tracks']['items']
    #print(items)
    
    if "user" in session:
        val = user_extract(email=session['user'])
        lan = user_extract_language(email=session['user'])
        name = str(lan) + " " + name + " song"
        

    # #    return render_template('prediction.html',user=lan)
    # else:
    #     return render_template('login.html')



    p=[]
    for i in range(len(items)):
        b=items[i]['id']
        p.append(b)
    import random
    items = random.sample(items, len(items))
    maxindex1="no emotion"

    return render_template('search1.html',
                           name=name,
                           results=items,
                           api_url=api_url, 
                           search_type=search_type)

def make_search2(search_type, name):
    global maxindex1
    data = spotify.search(name,limit=20,type="track")   
    api_url = data['tracks']['items']
    items = data['tracks']['items']
    #print(items)
    p=[]
    for i in range(len(items)):
        b=items[i]['id']
        p.append(b)
    import random
    items = random.sample(items, len(items))
    dst=os.listdir("static/uploadeddetected")
    k=dst[0]
    print(k)
    maxindex1="no emotion"

    return render_template('search2.html',
                           name=name,
                           results=items,
                           api_url=api_url,
                           k=k,
                           search_type=search_type)

# import mediapipe as mp




def dbConnection():
    connection = pymysql.connect(host="localhost", user="root", password="root", database="musicreco")
    return connection

def dbClose():
    dbConnection().close()
    return
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf

maxindex1="no emotion"

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt","res10_300x300_ssd_iter_140000.caffemodel")

UPLOAD_FOLDER = 'static/uploadedimages'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','wav','mp3'}











app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'random string'
@app.route('/')
def index():
    return render_template('index.html')


@app.route("/framehtml",methods=['POST','GET'])
def framehtml():
    if "user" in session:
        return render_template('frames.html')
    else:
        return render_template('login.html')
    #return render_template("frames.html")



@app.route("/imagecaptures",methods=['POST','GET'])
def imagecaptures():
    return render_template('capture.html')


    
    
@app.route("/imcaptured",methods=['POST','GET'])
def imcaptured():
    if request.method == "POST":
        
        import cv2
        import imutils
        import base64
        import json
        global maxindex1

        emotions=[]
        if len(emotions) !=0:
            emotions.clear()
            
        try:
            os.remove('static/uploadeddetected/uplimg.jpg')
        finally :
        
       


            data = request.get_json()['img']
            print("HELLOHUIHUIHDHDJWSBDXHBDCHJ")

            
            bytes(data, 'utf-8')
            fdata = data.split(',')[1]
            
            
            
            
            
            img = base64.b64decode(fdata)
            

            nparr = np.fromstring(img, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)    
            frame = imutils.resize(img, width=400)
            print(frame)

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
            
                # pass the blob through the network and obtain the detections and
                # predictions
            net.setInput(blob)
            detections = net.forward()
                #print(detections)
            faceBoxes = []
                # loop over the detections
            for i in range(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with the
                    # prediction
                confidence = detections[0, 0, i, 2]
                
            
                    # filter out detections by confidence
                if confidence < 0.7:
                    continue
            
                    # compute the (x, y)-coordinates of the bounding box for the
                    # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    #print(gray_frame)
                    #print(box)
                frameHeight = frame.shape[0]
                frameWidth = frame.shape[1]
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
                    #faceBoxes=box
                    #print(faceBoxes)
                    #print(faceBoxes[0][2])
                    
                roi_gray_frame = gray_frame[faceBoxes[0][2]:faceBoxes[0][2] + faceBoxes[0][3], faceBoxes[0][0]:faceBoxes[0][0] + faceBoxes[0][3]]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                emotion_dict = {0: "Angry", 1: "Fearful", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprised"}
            
                emotionlistforadding=['Angry','Fearful','Happy','Neutral','Sad','Surprised']
                    #break
                from tensorflow.keras.models import model_from_json
                model = model_from_json(open("fer.json", "r").read())
                    #load weights
                model.load_weights('fer.h5')
                emotion_prediction = model.predict(cropped_img)
                    #print(emotion_prediction)
                maxindex = int(np.argmax(emotion_prediction))
                
                maxindex1=emotion_dict[maxindex]
                labelemotion= "{}".format(emotion_dict[maxindex])
                emotions.append(labelemotion)
                
                    
                    
                    # draw the bounding box of the face along with the associated
                    # probability
                    #text = "{:.2f}%".format(confidence * 100)

            
                    
                
                    
                    
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
                cv2.putText(frame, labelemotion, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 255, 255), 2, cv2.LINE_AA)
                try:
                    os.remove('static/uploadeddetected/uplimg.jpg')

                finally:
                    cv2.imwrite("static/uploadeddetected/uplimg"+".jpg", frame)
                    
                
                return render_template('newtest2.html')
       
        

    # else :
        
    #      return render_template('newtest.html')

        # import collections 
        # from collections import Counter
        # occurrences3 = collections.Counter(emotions)
        # if(collections.Counter(emotions)=={}):
        #     #  flash('Emotion not detected')
        #     search_type="track"
        #     name="party"
        #     return make_search(search_type, name)
        # else:
        #     print(emotions)
        #     print(occurrences3)
        #     Keymax_emotion = max(zip(occurrences3.values(), occurrences3.keys()))[1]
            
        #     #  Keymax_emotion=Keymax_emotion[10:]
            
            
        
        #     print(Keymax_emotion)
        #     if Keymax_emotion=="Happy":
        #         search_type="track"
        #         name="party"
        #         return make_search(search_type, name)
        #     elif Keymax_emotion=="Angry":
        #         search_type="track"
        #         name="sad"
        #         return make_search(search_type, name)
        #     elif Keymax_emotion=="Disgusted":
        #         search_type="track"
        #         name="motivational"
        #         return make_search(search_type, name)
        #     elif Keymax_emotion=="Fearful":
        #         search_type="track"
        #         name="motivational"
        #         return make_search(search_type, name)
        #     elif Keymax_emotion=="Neutral":
        #         search_type="track"
        #         name="party"
        #         return make_search(search_type, name)
        #     elif Keymax_emotion=="Sad":
        #         print("HI")
        #         search_type="track"
        #         name="sad"
        #         return make_search(search_type, name)
        #     else:
        #         search_type="track"
        #         name="surprised"
        #         return make_search(search_type, name)
                    
                    
                    
        #         msg="Some Camera Incompatibility Issue"
        #         return msg




# @app.route('/searchQuery', methods=['POST'])
# def searchQuery():
#     name = request.form.get("q")
#     search_type= "track"
#     # search_type=request.form['search_type']
#     # name=request.form.get("")
#     # print(search_type)
#     # print(name)
#     return make_search(search_type, name)






@app.route("/uploadimage",methods=['POST','GET'])
def uploadimage():
    if request.method == "POST":
        import imutils
        emotions=[]
        file = request.files['email']
        from werkzeug.utils import secure_filename
        from werkzeug.datastructures import  FileStorage
        filename = secure_filename(file.filename)
        print("inside uploadimage function")
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv2.imread("static/uploadedimages/"+str(filename))
        frame = imutils.resize(img, width=400)
 
     # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
     
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        #print(detections)
        faceBoxes = []
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
    
            # filter out detections by confidence
            if confidence < 0.7:
                continue
    
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #print(gray_frame)
            #print(box)
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            #faceBoxes=box
            #print(faceBoxes)
            #print(faceBoxes[0][2])
            
            roi_gray_frame = gray_frame[faceBoxes[0][2]:faceBoxes[0][2] + faceBoxes[0][3], faceBoxes[0][0]:faceBoxes[0][0] + faceBoxes[0][3]]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    
            emotionlistforadding=['Angry','Disgusted','Fearful','Happy','Neutral','Sad','Surprised']
            #break
            import tensorflow as tf
            #new_model = tf.keras.models.load_model('emotion_model.h5')
            from tensorflow.keras.models import model_from_json
            model = model_from_json(open("fer.json", "r").read())
            #load weights
            model.load_weights('fer.h5')
            from tensorflow import keras
            #model=keras.models.load_model("model.h5")
            emotion_prediction = model.predict(cropped_img)
            #print(emotion_prediction)
            maxindex = int(np.argmax(emotion_prediction))
            print(maxindex)
            labelemotion= "{}".format("" + emotion_dict[maxindex])
            emotions.append(labelemotion)
            
             
            # draw the bounding box of the face along with the associated
            # probability
            #text = "{:.2f}%".format(confidence * 100)
            
            
            
            
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(frame, labelemotion, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 255, 255), 2, cv2.LINE_AA)
           
            cv2.imwrite("static/uploadeddetected/uplimg"+".jpg", frame)
    import collections 
    from collections import Counter
    occurrences3 = collections.Counter(emotions)
    #print(occurrences3)
    if(collections.Counter(emotions)=={}):
        flash('Emotion not detected')
        return redirect(url_for('framehtml'))
    else:

        Keymax_emotion = max(zip(occurrences3.values(), occurrences3.keys()))[1]
        # if(Keymax_emotion==''):
        #     flash('Emotion not detected')
        #     return redirect(url_for('frames'))
        print(Keymax_emotion)
        Keymax_emotion=Keymax_emotion[10:]
        print(Keymax_emotion)
        #return render_template("showemotion.html", Keymax_emotion=Keymax_emotion)

        if Keymax_emotion=="Happy": #gfhjgkhgfvhftyfvughtfvghfvhgfvhgfvhgfvhgfvhgfvhgfvhgfvhgfvhgfvhgfvhgfvhgfvghfvhgfvhgv
            search_type="track"
            name="happy "
            return make_search2(search_type, name)
        elif Keymax_emotion=="Angry":
            search_type="track"
            name="classical "
            return make_search2(search_type, name)
        elif Keymax_emotion=="Disgusted":
            search_type="track"
            name="motivational"
            return make_search2(search_type, name)
        elif Keymax_emotion=="Fearful":
            search_type="track"
            name="calming "
            return make_search2(search_type, name)
        elif Keymax_emotion=="Neutral":
            search_type="track"
            name="party"
            return make_search2(search_type, name)
        elif Keymax_emotion=="Sad":
            search_type="track"
            name="motivational slow"
            return make_search2(search_type, name)
        else:
            search_type="track"
            name="happy"
            return make_search2(search_type, name)
        
        
        
    
    #return render_template('frames.html')



@app.route("/question_new",methods=['POST','GET'])
def question_new():
    if request.method == "POST":
        name = request.form.get('mood2')
        
        if name=="Romantic" :
            search_type="track"
            name="romantic hindi song"
            return make_search(search_type, name)
        if name=="Demotivated":
            search_type="track"
            name = "relaxing hindi song"
            return make_search(search_type, name)
        if name=="Happy":
            name2=request.form.get('celeb1')
            if name2=="Yes" or name2=="yes":
                search_type="track"
                name="happy uplifting songs"
                return make_search(search_type, name)
            else:
                 search_type="track"
                 name="lonely"
                 return make_search(search_type, name)
            name3=request.form.get('sooth')
            if name3=="Yes" or name3=="yes":
                search_type="track"
                name="ragas"
                return make_search(search_type, name)
            else:
                 search_type="track"
                 name="party"
                 return make_search(search_type, name)
        if name=="Sad":
            name2=request.form.get('lonely')
            if name2=="Yes" or name2=="yes":
                search_type="track"
                name="lonely"
                return make_search(search_type, name)
            else:
                 search_type="track"
                 name="party"
                 return make_search(search_type, name)
            name3=request.form.get('heartbroke')
            if name3=="Yes" or name3=="yes":
                search_type="track"
                name="heartbroke"
                return make_search(search_type, name)
            else:
                 search_type="track"
                 name="party"
                 return make_search(search_type, name)
            name4=request.form.get('sadness')
            if name4=="Yes" or name4=="yes":
                 search_type="track"
                 name="sad"
                 return make_search(search_type, name)
            else:
                 search_type="track"
                 name="happy"
                 return make_search(search_type, name)
        if name=="Angry":
            name2=request.form.get('celeb2')
            if name2=="Yes" or name2=="yes":
                search_type="track"
                name="classical"
                return make_search(search_type, name)
            else:
                 search_type="track"
                 name="motivational"
                 return make_search(search_type, name)
            name3=request.form.get('sooth2')
            if name3=="Yes" or name3=="yes":
                search_type="track"
                name="motivational"
                return make_search(search_type, name)
            else:
                 search_type="track"
                 name="relax"
                 return make_search(search_type, name)
        
        








@app.route("/callback/")
def callback():

    auth_token = request.args['code']
    auth_header = spotify.authorize(auth_token)
    session['auth_header'] = auth_header

    return profile()
@app.route('/profile')
def profile():
    if 'auth_header' in session:
        auth_header = session['auth_header']
        # get profile data
        profile_data = spotify.get_users_profile(auth_header)

        # get user playlist data
        playlist_data = spotify.get_users_playlists(auth_header)

        # get user recently played tracks
        recently_played = spotify.get_users_recently_played(auth_header)
        
        if valid_token(recently_played):
            return render_template("profile.html",
                               user=profile_data,
                               playlists=playlist_data["items"],
                               recently_played=recently_played["items"])

    return render_template('profile.html')

@app.route('/home.html')
def home():
    if "user" in session:
        return render_template('home.html')
    else:
        return render_template('login.html')
    # return render_template('home.html')

@app.route('/prediction',methods=['POST','GET'])
def prediction():
    if "user" in session:
        val = user_extract(email=session['user'])
        lan = user_extract_language(email=session['user'])
        return render_template('prediction.html',user=lan)
    else:
        return render_template('login.html')
    # return render_template('prediction.html')

@app.route('/captureimages',methods=['POST','GET'])
def captureimages():
    import cv2
    #from cv2 import *
    videoCaptureObject = cv2.VideoCapture(0)
    i=0
    while(i<=9):
        ret,frame = videoCaptureObject.read()
        cv2.imwrite("static/captureimages/NewPicture"+str(i)+".jpg",frame)
        i=i+1
        
    videoCaptureObject.release()
    cv2.destroyAllWindows()
    return render_template('prediction.html')



@app.route('/question',methods=['POST','GET'])
def question():
    if request.method == "POST":


      
            

        mood  =  request.form.get("selectop")
        
        name  = request.form.get('happyrad1')
        name2 = request.form.get('happyrad2')
        print("--------------------------------")
        print(mood)
        print(name)
        print(name2)
        if name=="Yes":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            print("line 741")
            name="Happy uplifting"#+str(res)
            return make_search(search_type, name)
        if name=="No" and name2=="No":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Calming slow songs "#+str(res)
            return make_search(search_type, name)
        if name=="No" and name2=="Yes":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Calm romantic "#+str(res)
            return make_search(search_type, name)
        sadname3 = request.form.get('sadrad1')
        sadname4 = request.form.get('sadrad2')
        sadname5 = request.form.get('sadrad3')
        if sadname3=="Yes":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Motivational  "#+str(res)
            return make_search(search_type, name)
        if sadname3=="No" and sadname4=="Yes":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Romantic slow sad songs "#+str(res)
            return make_search(search_type, name)
        if sadname3=="No" and sadname4=="No":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Soft songs calming"#+str(res)
            return make_search(search_type, name)
        if sadname3=="No" and sadname4=="No" and sadname5=="Yes":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="soothing songs"#+str(res)
            return make_search(search_type, name)
        if sadname3=="No" and sadname4=="No" and sadname5=="No":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Sad calming"#+str(res)
            return make_search(search_type, mood)
        
        angryname6 = request.form.get('angryrad1')
        angryname7 = request.form.get('angryrad2')
        if angryname6=="Yes":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Calm romantic"#+str(res)
            return make_search(search_type, name)
        if angryname6=="No" and angryname7=="Yes":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Motivaltional songs"#+str(res)
            return make_search(search_type, name)
        if angryname6=="No" and angryname7=="No":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Motivational upbeat"
            return make_search(search_type, name)
            
                
        
        
        
        
        
        
        
        if mood=="Romantic":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="romantic hindi"
            return make_search(search_type, name)
        if mood=="Demotivated":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="motivational slow"
            return make_search(search_type, name)
        # else:
        #     search_type="track"
        #     # con = dbConnection()
        #     # cursor = con.cursor()
        #     # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
        #     # res = cursor.fetchone()
        #     # res=res[0]
        #     # print(res)
        #     name="motivational uplifting "
        #     return make_search(search_type, name)
        
        # name = request.form.get('mood')
        # if name=="Romantic":
        #     search_type="track"
        #     name="romantic hindi songs"
        #     return make_search(search_type, name)
        # if name=="Demotivated":
        #     search_type="track"
        #     name = "relaxing hindi songs"
        #     return make_search(search_type, name)
        # if name=="Happy":
        #     name2=request.form.get('celeb1')
        #     if name2=="Yes" or name2=="yes":
        #         search_type="track"
        #         name="happy uplifting songs"
        #         return make_search(search_type, name)
        #     else:
        #          search_type="track"
        #          name="happy soothing songs"
        #          return make_search(search_type, name)
        #     name3=request.form.get('sooth')
        #     if name3=="Yes" or name3=="yes":
        #         search_type="track"
        #         name="ragas"
        #         return make_search(search_type, name)
        #     else:
        #          search_type="track"
        #          name="party"
        #          return make_search(search_type, name)
        # if name=="Sad":
        #     name2=request.form.get('lonely')
        #     name3=request.form.get('heartbroke')
        #     name4=request.form.get('sadness')
        #     if name2=="Yes" or name2=="yes" :
        #         search_type="track"
        #         name="slow calming songs"
        #         return make_search(search_type, name)
        #     else:
        #          search_type="track"
        #          name="calming songs uplifting"
        #          return make_search(search_type, name)
        #     #name3=request.form.get('heartbroke')
        #     if name3=="Yes" or name3=="yes":
        #         search_type="track"
        #         name=""
        #         return make_search(search_type, name)
        #     else:
        #          search_type="track"
        #          name="party"
        #          return make_search(search_type, name)
        #     name4=request.form.get('sadness')
        #     if name4=="Yes" or name4=="yes":
        #          search_type="track"
        #          name="calming uplifting songs"
        #          return make_search(search_type, name)
        #     else:
        #          search_type="track"
        #          name="soothing calming songs"
        #          return make_search(search_type, name)
        # if name=="Angry":
        #     name2=request.form.get('celeb2')
        #     if name2=="Yes" or name2=="yes":
        #         search_type="track"
        #         name="classical music calming"
        #         return make_search(search_type, name)
        #     else:
        #          search_type="track"
        #          name="motivational songs uplifting"
        #          return make_search(search_type, name)
        #     name3=request.form.get('sooth2')
        #     if name3=="Yes" or name3=="yes":
        #         search_type="track"
        #         name="motivational"
        #         return make_search(search_type, name)
        #     else:
        #          search_type="track"
        #          name="relax"
        #          return make_search(search_type, name)

            
        #if name==""
        
        #return render_template('question.html')
    return render_template('newtest.html')


@app.route('/question2',methods=['POST','GET'])
def question2():
    if request.method == "POST":

        print(maxindex1)
        if maxindex1 == "no emotion":
            flash('Emotion not detected')
            print("inside")
            return redirect(url_for('framehtml'))


        moodyes = request.form.get('moodyes')
        print("outSIDE IF STATEMENT 961")
        print(moodyes)
        if moodyes == "moodyes":
            print("INSIDE IF STATEMENT 961")
            search_type="track"
            mood= maxindex1
            print(mood)
            if mood=="Happy":
                name="happy "
                return make_search(search_type, name)
            if mood=="Romantic":
                name="romantic calming"
                return make_search(search_type, name)
            if mood=="Demotivated":
                name="motivational slow"
                return make_search(search_type, name)
            if mood=="Sad":
                name="calming "
                return make_search(search_type, name)
            if mood=="Angry":
                name="classical "
                return make_search(search_type, name)

            # return make_search2(search_type,mood)

        

        mood  =  request.form.get("selectop")
        name  = request.form.get('happyrad1')
        name2 = request.form.get('happyrad2')
        print("--------------------------------")
        print(mood)
        print(name)
        print(name2)
        if name=="Yes":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Happy "#+str(res)
            return make_search(search_type, name)
        if name=="No" and name2=="No":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Calming slow "#+str(res)
            return make_search(search_type, name)
        if name=="No" and name2=="Yes":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Calm romantic "#+str(res)
            return make_search(search_type, name)
        sadname3 = request.form.get('sadrad1')
        sadname4 = request.form.get('sadrad2')
        sadname5 = request.form.get('sadrad3')
        if sadname3=="Yes":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Slow romantic "#+str(res)
            return make_search(search_type, name)
        if sadname3=="No" and sadname4=="Yes":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Romantic slow sad songs "#+str(res)
            return make_search(search_type, name)
        if sadname3=="No" and sadname4=="No":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Soft calming"#+str(res)
            return make_search(search_type, name)
        if sadname3=="No" and sadname4=="No" and sadname5=="Yes":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Calming "#+str(res)
            return make_search(search_type, name)
        if sadname3=="No" and sadname4=="No" and sadname5=="No":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Calming and soothing"#+str(res)
            return make_search(search_type, name)
        angryname6 = request.form.get('angryrad1')
        angryname7 = request.form.get('angryrad2')
        if angryname6=="Yes":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Calm romantic"#+str(res)
            return make_search(search_type, name)
        if angryname6=="No" and angryname7=="Yes":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Motivational songs"#+str(res)
            return make_search(search_type, name)
        if angryname6=="No" and angryname7=="No":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="Calming"
            return make_search(search_type, name)
              
                 
        
        
        
        
        
        
        
        if mood=="Romantic":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="romantic "
            return make_search(search_type, name)
        if mood=="Demotivated":
            search_type="track"
            # con = dbConnection()
            # cursor = con.cursor()
            # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
            # res = cursor.fetchone()
            # res=res[0]
            # print(res)
            name="motivational slow upbeat "
            return make_search(search_type, name)
        # else:
        #     search_type="track"
        #     # con = dbConnection()
        #     # cursor = con.cursor()
        #     # cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
        #     # res = cursor.fetchone()
        #     # res=res[0]
        #     # print(res)
        #     name="motivational uplifting "
        #     return make_search(search_type, name)
        
        # name = request.form.get('mood')
        # if name=="Romantic":
        #     search_type="track"
        #     name="romantic hindi songs"
        #     return make_search(search_type, name)
        # if name=="Demotivated":
        #     search_type="track"
        #     name = "relaxing hindi songs"
        #     return make_search(search_type, name)
        # if name=="Happy":
        #     name2=request.form.get('celeb1')
        #     if name2=="Yes" or name2=="yes":
        #         search_type="track"
        #         name="happy uplifting songs"
        #         return make_search(search_type, name)
        #     else:
        #          search_type="track"
        #          name="happy soothing songs"
        #          return make_search(search_type, name)
        #     name3=request.form.get('sooth')
        #     if name3=="Yes" or name3=="yes":
        #         search_type="track"
        #         name="ragas"
        #         return make_search(search_type, name)
        #     else:
        #          search_type="track"
        #          name="party"
        #          return make_search(search_type, name)
        # if name=="Sad":
        #     name2=request.form.get('lonely')
        #     name3=request.form.get('heartbroke')
        #     name4=request.form.get('sadness')
        #     if name2=="Yes" or name2=="yes" :
        #         search_type="track"
        #         name="slow calming songs"
        #         return make_search(search_type, name)
        #     else:
        #          search_type="track"
        #          name="calming songs uplifting"
        #          return make_search(search_type, name)
        #     #name3=request.form.get('heartbroke')
        #     if name3=="Yes" or name3=="yes":
        #         search_type="track"
        #         name=""
        #         return make_search(search_type, name)
        #     else:
        #          search_type="track"
        #          name="party"
        #          return make_search(search_type, name)
        #     name4=request.form.get('sadness')
        #     if name4=="Yes" or name4=="yes":
        #          search_type="track"
        #          name="calming uplifting songs"
        #          return make_search(search_type, name)
        #     else:
        #          search_type="track"
        #          name="soothing calming songs"
        #          return make_search(search_type, name)
        # if name=="Angry":
        #     name2=request.form.get('celeb2')
        #     if name2=="Yes" or name2=="yes":
        #         search_type="track"
        #         name="classical music calming"
        #         return make_search(search_type, name)
        #     else:
        #          search_type="track"
        #          name="motivational songs uplifting"
        #          return make_search(search_type, name)
        #     name3=request.form.get('sooth2')
        #     if name3=="Yes" or name3=="yes":
        #         search_type="track"
        #         name="motivational"
        #         return make_search(search_type, name)
        #     else:
        #          search_type="track"
        #          name="relax"
        #          return make_search(search_type, name)

            
        #if name==""
        
        #return render_template('question.html')
        
       
    return render_template('newtest2.html',maxind = maxindex1)




@app.route('/searchPage', methods=['GET'])
def searchPage():
    # if request.method == "GET":
    #     name = request.form.get("songName")
    #     search_type= "track"
    #     print(songName)
    #     # search_type=request.form['search_type']
    #     # name=request.form.get("")
    #     print(search_type)
    #     print(name)
    #     return make_search(search_type, name)
    if "user" in session:
        return render_template('searchPage.html')
    else:
        return render_template('login.html')
    # return render_template('searchPage.html')

@app.route('/songResults',methods = ['POST'])
def songResults():
    # try:

    status=""
    songName = request.form.get("songName")
    print(songName)
    if request.method == "POST":
        search_type="track"
        print(search_type)
        print(songName)
        return make_search_own(search_type, songName)

    return render_template('songResults.html')
    

@app.route('/register',methods=['POST','GET'])
def register():
    return render_template('register.html')
    
@app.route('/register_response',methods=['POST'] )
def register_response():
    # try:
    status=""
    fname = request.form.get("name")
    lang = request.form.get("language")
    # add = request.form.get("add")
    # pno = request.form.get("pno")
    email = request.form.get("email")
    passe = request.form.get("pass")
    pass1 =  request.form.get("pass1")
    if passe==pass1:
        dic = {'name':fname,'email':email,'language':lang}
        data_upload(dic)
        val=admin_signup1(email, pass1)
        print(val)
        # status="success"
        if val:
            return render_template('login.html')
        else:
            flash("Email already exists")
            return render_template('register.html')
    else:
        # status="fail"
        flash("Please Enter Same Password")
        return render_template('register.html')
    # dic = {'name':fname,'address':add,'phone':pno,'email':email}
    # data_upload(dic)
    # val=admin_signup1(email, pass1)
    # return render_template('login.html')
#         if not res:
#             sql = "INSERT INTO userdetailes (name, address,phone,email,password) VALUES (%s,%s, %s, %s, %s)"
#             val = (fname ,add ,pno ,email ,pass1)
#             print(sql," ",val)
#             cursor.execute(sql, val)
#             con.commit()
#             status= "success"
#             return render_template("login.html")
#         else:
#             status = "Already available"
#         #return status
#         return redirect(url_for('index'))
    # except Exception as e:
    #     print(e)
    #     print("Exception occured at user registration")
    #     return redirect(url_for('index'))
    #     finally:
    #         dbClose()
    # return render_template('register.html')

@app.route('/login',methods=['POST','GET'])
def login():
    return render_template('login.html')

@app.route('/login_response',methods=['POST'])
def login_response():
    msg = ''
    # if request.method == "POST":
    # session.pop('user',None)
    mailid = request.form.get("email")
    password = request.form.get("pass1")
    # user = User.query.filter_by(email=email).first()
    val = admin_login1(mailid, password)
    if val :
        session['user'] = mailid
        return render_template("home.html")
    # elif user:
    #     msg = "User not found"
    #     flash(msg)
    #     return render_template("login.html")
    else:
        # print(result_count)
        msg = 'Incorrect username/password!'
        flash(msg)
        # print(msg)
        # return msg
        return redirect(url_for('login'))



@app.route('/logout',methods=['POST'])
def logout():
    print("Logged Out")

    session.pop('logged_in',None)
    # val = a
    # session.pop('user',None)
    return render_template('index.html')



@app.route('/project.html')
def contact():
    if "user" in session:
        return render_template('project.html')
    else:
        return render_template('login.html')
    # return render_template('project.html')
@app.route('/analysis.html')
# def analysis():
#    if "user" in session:
#         return render_template('analysis.html')
#     else:
#         return render_template('login.html')
#    #return render_template('analysis.html')
@app.route('/modification.html')
def Modification():
    return render_template('modification.html')


if __name__=="__main__":
    # app.run("0.0.0.0")
    app.run(debug=True,host="0.0.0.0")