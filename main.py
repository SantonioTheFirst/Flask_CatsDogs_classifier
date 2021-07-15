# print('Hello!')
import numpy as np
import tensorflow

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout

from flask import Flask, render_template, request, url_for
# from PIL import Image

import warnings
warnings.filterwarnings("ignore")

image_size = 299

def create_model():
    Xception_model = tensorflow.keras.applications.xception.Xception(
        include_top=False, input_shape=(image_size, image_size, 3), pooling='avg'
    )
    Xception_model.trainable = False
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=Xception_model.output_shape[1:]))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.load_weights('model_weights')
    final_model = Model(inputs=Xception_model.input, outputs=model(Xception_model.output))
    final_model.compile(loss='categorical_crossentropy',
     optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-5), metrics=['accuracy'])
    return final_model


def get_image_array(img_path):
    img = tensorflow.keras.preprocessing.image.load_img(img_path, target_size=(image_size, image_size))
    img_arr = tensorflow.keras.preprocessing.image.img_to_array(img)
    img_arr /= 255
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr


def get_predictions(file_path):
    x = get_image_array(f'{file_path}')
    prediction = model.predict(x)
    return f'cat: {np.round(prediction[0][0] * 100, 4)}%\ndog: {np.round(prediction[0][1] * 100, 4)}%'


model = create_model()


app = Flask(__name__)

@app.route('/')
def main_page():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No files!'
    else:
        print('Inside else')
        file = request.files['file']
        file.save(f'images/{file.filename}')
        # img = Image.open(f'images/{file.filename}')
        # new_height = 720
        # new_width = int(new_height / img.width * img.height)
        # img.resize((new_width, new_height))
        # img.save(f'images/{file.filename}')
        prediction = get_predictions(f'images/{file.filename}')
        print(prediction)
        return render_template('index.html', param=prediction)


# @app.route('/model/<file_path>')



if __name__ == '__main__':
    app.run(debug=True)



# print('The end')
