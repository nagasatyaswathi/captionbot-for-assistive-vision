import numpy as np
import os
from flask import Flask,flash, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
from gtts import gTTS
from IPython.display import Audio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import ssl
import matplotlib
matplotlib.use('Agg')

ssl._create_default_https_context = ssl._create_unverified_context

UPLOAD_FOLDER = 'C:/Users/KOTE/Desktop/Raghu/archive/static/photo'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(filename, model):
        try:
            image = Image.open(filename)

        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4:
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return render_template('index.html', prediction_text='File Upload Failed...')
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return render_template('index.html', prediction_text='No File Selected...')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_path = 'C:/Users/KOTE/Desktop/Raghu/archive/static/photo/{}'.format(filename)

    max_length = 32
    tokenizer = load(open("C:/Users/KOTE/Desktop/Raghu/archive/tokenizer.p","rb"))
    model = load_model('C:/Users/KOTE/Desktop/Raghu/archive/models/model_9.h5')
    xception_model = Xception(include_top=False, pooling="avg")

    photo = extract_features(img_path, xception_model)
    img = Image.open(img_path)

    description = generate_desc(model, tokenizer, photo, max_length)
    print("\n\n")
    plt.imshow(img)
    stopwords = ['start', 'end']
    querywords = description.split()

    resultwords  = [word for word in querywords if word.lower() not in stopwords]
    result = ' '.join(resultwords)

    print(result)
    audio_filename= filename.replace(".", "_")
    tts = gTTS(text=result, lang='en',slow=False)
    tts.save('C:/Users/KOTE/Desktop/Raghu/archive/static/audio/{}.mp3'.format(audio_filename))
    Audio('C:/Users/KOTE/Desktop/Raghu/archive/static/audio/{}.mp3'.format(audio_filename), autoplay=True)
    full_filename = 'http://127.0.0.1:5000/static/photo/{}'.format(filename)
    audio_path= 'http://127.0.0.1:5000/static/audio/{}.mp3'.format(audio_filename)
    print(full_filename)
    return render_template('index.html', user_image = full_filename ,audio_file= audio_path, prediction_text='Caption : {}'.format(result))


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)
