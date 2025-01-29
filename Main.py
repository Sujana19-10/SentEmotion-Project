from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import filedialog
import os
import cv2
import soundfile
import librosa
from tensorflow.keras.utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Convolution2D
from keras.models import Sequential, model_from_json
import pickle
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

main = tkinter.Tk()
main.title("Sentiment analysis using Machine Learning on multiple modalities")  # designing main screen
main.geometry("1300x1200")

sid = SentimentIntensityAnalyzer()

# Define global variables
global filename
global X, Y
global face_classifier
global speech_X, speech_Y
global speech_classifier

# Define emotion lists
face_emotion = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
speech_emotion = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Define 'names' variable
names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']  # Example names for emotions

def getID(name):
    index = 0
    for i in range(len(names)):
        if names[i] == name:
            index = i
            break
    return index

def upload():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n")

def processDataset():
    text.delete('1.0', END)
    global X, Y
    global speech_X, speech_Y
    '''
    X = []
    Y = []
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            print(name+" "+root+"/"+directory[j])
            if 'Thumbs.db' not in directory[j]:
                img = cv2.imread(root+"/"+directory[j])
                img = cv2.resize(img, (32,32))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(32,32,3)
                X.append(im2arr)
                Y.append(getID(name))
        
    X = np.asarray(X)
    Y = np.asarray(Y)
    print(Y)

    X = X.astype('float32')
    X = X/255    
    test = X[3]
    test = cv2.resize(test,(400,400))
    cv2.imshow("aa",test)
    cv2.waitKey(0)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    np.save('model/X.txt',X)
    np.save('model/Y.txt',Y)
    '''
    
    X = np.load('model/speechX.txt.npy')
    Y = np.load('model/speechY.txt.npy')
    speech_X = np.load('model/speechX.txt.npy')
    speech_Y = np.load('model/speechY.txt.npy')
    text.insert(END,"Total number of images found in dataset is  : "+str(len(X))+"\n")
    text.insert(END,"Total facial expression found in dataset is : "+str(face_emotion)+"\n")
    text.insert(END,"Total number of speech emotion audio files found in dataset is  : "+str(speech_X.shape[0])+"\n")
    text.insert(END,"Total speech emotion found in dataset is : "+str(speech_emotion)+"\n")
    

def trainSpeechCNN():
    global speech_classifier
    if os.path.exists('model/speechmodel.json'):
        with open('model/speechmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            speech_classifier = model_from_json(loaded_model_json)
        speech_classifier.load_weights("model/speech_weights.h5")
    else:
        speech_classifier = Sequential()
        speech_classifier.add(Convolution2D(32, 1, 1, input_shape=(speech_X.shape[1], speech_X.shape[2], speech_X.shape[3]), activation='relu'))
        speech_classifier.add(MaxPooling2D(pool_size=(1, 1)))
        speech_classifier.add(Convolution2D(32, 1, 1, activation='relu'))
        speech_classifier.add(MaxPooling2D(pool_size=(1, 1)))
        speech_classifier.add(Flatten())
        speech_classifier.add(Dense(units=256, activation='relu'))
        speech_classifier.add(Dense(units=speech_Y.shape[1], activation='softmax'))
        
        speech_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = speech_classifier.fit(speech_X, speech_Y, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
        speech_classifier.save_weights('model/speech_weights.h5')            
        model_json = speech_classifier.to_json()
        with open("model/speechmodel.json", "w") as json_file:
            json_file.write(model_json)

        with open('model/speechhistory.pckl', 'wb') as f:
            pickle.dump(hist.history, f)

    with open('model/speechhistory.pckl', 'rb') as f:
        data = pickle.load(f)
    acc = data['accuracy']
    accuracy = acc[-1] * 100  # Last epoch accuracy
    text.insert(END, f"CNN Speech Emotion Training Model Accuracy = {accuracy}%\n\n")

def trainFaceCNN():
    global face_classifier
    if os.path.exists('model/cnnmodel.json'):
        with open('model/cnnmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            face_classifier = model_from_json(loaded_model_json)
        face_classifier.load_weights("model/cnnmodel_weights.h5")
    else:
        face_classifier = Sequential()
        face_classifier.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3), activation='relu'))
        face_classifier.add(MaxPooling2D(pool_size=(2, 2)))
        face_classifier.add(Convolution2D(32, 3, 3, activation='relu'))
        face_classifier.add(MaxPooling2D(pool_size=(2, 2)))
        face_classifier.add(Flatten())
        face_classifier.add(Dense(units=256, activation='relu'))
        face_classifier.add(Dense(units=7, activation='softmax'))

        face_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = face_classifier.fit(X, Y, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
        face_classifier.save_weights('model/cnnmodel_weights.h5')            
        model_json = face_classifier.to_json()
        with open("model/cnnmodel.json", "w") as json_file:
            json_file.write(model_json)

        with open('model/facialhistory.pckl', 'wb') as f:
            pickle.dump(hist.history, f)

    with open('model/facialhistory.pckl', 'rb') as f:
        data = pickle.load(f)
    acc = data['accuracy']
    accuracy = acc[-1] * 100  # Last epoch accuracy
    text.insert(END, f"CNN Face Emotion Training Model Accuracy = {accuracy}%\n\n")
 

def predictFaceExpression():
    global face_classifier
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (32,32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = face_classifier.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Facial Expression Recognized as : '+face_emotion[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Facial Expression Recognized as : '+face_emotion[predict], img)
    cv2.waitKey(0)



def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    sound_file.close()        
    return result

def predictSpeechExpression():
    global speech_classifier
    filename = filedialog.askopenfilename(initialdir="testSpeech")
    fname = os.path.basename(filename)
    test = []
    mfcc = extract_feature(filename, mfcc=True, chroma=True, mel=True)
    test.append(mfcc)
    test = np.asarray(test)
    test = test.astype('float32')
    test = test/255

    test = test.reshape((test.shape[0],test.shape[1],1,1))
    predict = speech_classifier.predict(test)
    predict = np.argmax(predict)
    print(predict)
    emotion = speech_emotion[predict-1]
    text.delete('1.0', END)
    text.insert(END,"Upload speech file : "+fname+" Emotion Recognized as : "+emotion+"\n") 
    


def graph():
    f = open('model/cnnhistory.pckl', 'rb')
    cnn_data = pickle.load(f)
    f.close()
    face_accuracy = cnn_data['accuracy']
    face_loss = cnn_data['loss']

    f = open('model/speechhistory.pckl', 'rb')
    cnn_data = pickle.load(f)
    f.close()
    speech_accuracy = cnn_data['accuracy']
    speech_loss = cnn_data['loss']
    sa = []
    sl = []
    for i in range(90,100):
        sa.append(speech_accuracy[i])
        sl.append(speech_loss[i])

    fa = []
    fl = []
    for i in range(20,30):
        fa.append(face_accuracy[i])
        fl.append(face_loss[i])

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Accuracy')
    plt.plot(fa, 'ro-', color = 'green')
    plt.plot(fl, 'ro-', color = 'orange')
    plt.plot(sa, 'ro-', color = 'blue')
    plt.plot(sl, 'ro-', color = 'red')
    plt.legend(['Face Emotion Accuracy', 'Face Emotion Loss','Speech Emotion Accuracy','Speech Emotion Loss'], loc='upper left')
    plt.title('CNN Face & Speech Emotion Accuracy Comparison Graph')
    plt.show()

def textEmotion():
    text.delete('1.0', END)
    sentence = tf1.get()
    sentiment_dict = sid.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    positive = sentiment_dict['pos']
    neutral = sentiment_dict['neu']
    compound = sentiment_dict['compound']
    result = ''
    if compound >= 0.05: 
        result = 'Happy'   
    elif compound <= -0.05 : 
        result = 'Disgusted'
    elif compound >= 0.03 and compound < -0.05: 
        result = 'Sad'
    elif compound >= 0.01 and compound < 0.03: 
        result = 'Fearful'      
    else : 
        result = 'Neutral'
    text.insert(END,"Entered Text: "+sentence+"\n\n")
    text.insert(END,"Predicted Emotion : "+str(result))
    tf1.delete(first=0,last=500)
    

font = ('times', 13, 'bold')
title = Label(main, text='Sentiment analysis using Machine Learning on multiple modalities')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=480,y=100)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Facial Emotion Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

cnnButton = Button(main, text="Train Facial Emotion CNN Algorithm", command=trainFaceCNN)
cnnButton.place(x=50,y=200)
cnnButton.config(font=font1) 

rnnButton = Button(main, text="Train Speech Emotion CNN Algorithm", command=trainSpeechCNN)
rnnButton.place(x=50,y=250)
rnnButton.config(font=font1) 

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)

predictfaceButton = Button(main, text="Predict Facial Emotion", command=predictFaceExpression)
predictfaceButton.place(x=50,y=350)
predictfaceButton.config(font=font1)

predictspeechButton = Button(main, text="Predict Speech Emotion", command=predictSpeechExpression)
predictspeechButton.place(x=50,y=400)
predictspeechButton.config(font=font1)

l1 = Label(main, text='Enter Sentence:')
l1.config(font=font1)
l1.place(x=50,y=450)

tf1 = Entry(main,width=80)
tf1.config(font=font1)
tf1.place(x=50,y=500)

exitButton = Button(main, text="Predict Emotion from Text", command=textEmotion)
exitButton.place(x=50,y=550)
exitButton.config(font=font1) 

main.config(bg='OliveDrab2')
main.mainloop()
