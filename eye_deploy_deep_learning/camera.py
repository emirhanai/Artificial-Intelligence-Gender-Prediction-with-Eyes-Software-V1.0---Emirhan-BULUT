from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K

# defining face detector
classifier=cv2.CascadeClassifier('/content/haarcascade_eye.xml.xml')
size = 4
labels_dict={0:'Female',1:'Male'}
color_dict={0:(0,255,0),1:(0,0,255)}
global loaded_model
graph1 = tf.compat.v1.Graph()
with graph1.as_default():
	session1 = tf.compat.v1.Session(graph=graph1)

	with session1.as_default():
		loaded_model = loaded_model = load_model('emirha_eye_predict.h5')
class VideoCamera(object):
    def __init__(self):
        # capturing video
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # releasing camera
        self.video.release()


    def get_frame(self):
        # extracting frames
        (rval, im) = self.video.read()
        im = cv2.flip(im, 1, 1)
        mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
        faces = classifier.detectMultiScale(im)
        for f in faces:
            (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
            #Save just the rectangle faces in SubRecFaces
            face_img = im[y:y+h, x:x+w]
            resized=cv2.resize(face_img,(256,256))
            normalized=resized/255.0
            reshaped = np.vstack([reshaped])
            K.set_session(session1)
            with graph1.as_default():
                results=loaded_model.predict(reshaped)
            if results[0][1] > results[0][0]:
                result = np.array([[1]])
            else:
                result = np.array([[0]])
            label = np.argmax(result)
            cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[result[label][0]],2)
            cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[result[label][0]],-1)
            cv2.putText(im, labels_dict[result[label][0]], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', im)
        return jpeg.tobytes()