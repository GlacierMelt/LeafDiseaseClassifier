import torch
from flask import Flask, request
from flask_cors import CORS
from flask import render_template, redirect, url_for, session, flash
from fastai.vision.all import *
from inference import get_prediction

import timm


app = Flask(__name__)
app.config["SECRET_KEY"] = 'TPmi4aLWRbyVq8zu9v82dWYW1'
cors = CORS(app)

model = timm.create_model('tf_efficientnet_b5_ns', num_classes=5)
model.load_state_dict(torch.load(
    Path('server/Efficient_B5_NS_Clean_best_tf_efficientnet_b5_ns.pt')))
model.eval()


@ app.route("/")
def index():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return render_template("index.html")


@ app.route('/login', methods=['POST'])
def login():
    if request.form['password'] == 'password' and request.form['username'] == 'admin':
        session['logged_in'] = True
    else:
        flash('wrong password!')
    return index()


@ app.route('/predict', methods=['GET', 'POST'])
def predict():
    img = PILImage.create(request.files['file'])
    probs, label = get_prediction(model, img)
    return f'{label} ({torch.max(torch.softmax(probs, 1)).item()*100:.0f}%)'


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
