from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)


model=load_model('model_mnist.h5')


model.make_predict_function()

def predict_label(img_path):
        i = image.load_img(img_path, target_size=(28,28), grayscale=True)
        i = image.img_to_array(i)
        resize_image= np.array([i], order='C')
        #img_out = image.array_to_img(resize_image.reshape(28,28,1))
        img_out.save(img_path)
        p = model.predict(resize_image)
        return p.argmax()

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Naturalis cloud project "

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
        if request.method == 'POST':
                img = request.files['my_image']

                img_path = "static/" + img.filename	
                img.save(img_path)

                p = predict_label(img_path)

        return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
    #app.debug = True
    app.run(host='0.0.0.0', port=8080, debug = True)