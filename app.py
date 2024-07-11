from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import librosa
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    
    # Read the uploaded .wav file
    y, sr = librosa.load(file, sr=None)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    
    # Compute mean across axis 0
    mel_mean = np.mean(mel, axis=0)
    
    s = pd.DataFrame([mel_mean])
    t = s.fillna(0)
    
    if t.shape[1] < 228:
        # Pad the data with zeros
        t = np.concatenate((t.values, np.zeros((1, 228 - t.shape[1]))), axis=1)
    elif t.shape[1] > 228:
        # Trim the data
        t = t.values[:, :228]

    features = t.reshape(1, 228, 1)
    
    prediction = model.predict(features)
    
    emotion = extract_emotion(prediction)
    
    return jsonify({'emotion': emotion})

def extract_emotion(prediction):
    # Get the class ID with the highest probability
    class_id = np.argmax(prediction)
    print(f'Class ID: {class_id}')
    
    if class_id == 0:
        emotion = 'angry'
    elif class_id == 1:
        emotion = 'disgust'
    elif class_id == 2:
        emotion = 'fear'
    elif class_id == 3:
        emotion = 'happy'
    elif class_id == 4:
        emotion = 'neutral'
    elif class_id == 5:
        emotion = 'surprise'
    elif class_id == 6:
        emotion = 'sad'
    else:
        emotion = 'Unknown'  
    
    return emotion


if __name__ == '__main__':
    app.run(debug=True)
