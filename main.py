from tkinter import filedialog
import customtkinter
import numpy as np
import os
import cv2 as cv
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

def Predict():

    model = models.load_model('nn')

    filepath = filedialog.askopenfilename()

    img = cv.imread(os.path.join(filepath))

    imagem = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    resize = tf.image.resize(imagem, (256, 256))

    img_array = np.expand_dims(resize / 255, 0)

    predict_test = model.predict(img_array)

    print(predict_test[0][0])

    if predict_test[0][0] > 0.5:
        g_title = 'Thats not nick Cage !!!'
    else:
        g_title = 'Thats actually Nick Cage >:D ! ! !'

    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title(g_title)
    plt.show()

def train_model():

    dataset = tf.keras.utils.image_dataset_from_directory("Images")

    data = dataset.map(lambda x, y: (x/255, y))

    train_set = data.take(round(len(data)*0.8))
    val_set = data.skip(round(len(data)*0.8)).take(round(len(data)*0.2))

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, (3, 3), 1, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, (3, 3), 1, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))

    model.compile('adam', loss = tf.losses.BinaryCrossentropy(), metrics = ['accuracy'])

    model.fit(train_set, epochs = 20, validation_data = val_set)

    model.save('nn')

customtkinter.set_appearance_mode('dark')
customtkinter.set_default_color_theme('dark-blue')

app = customtkinter.CTk()
app.geometry('425x275')
app.title('Nick Cage Identifier')
app.resizable(False, False)

frame = customtkinter.CTkFrame(master = app)
frame.pack(pady = 20, padx = 60, fill = 'both' , expand = True)

label = customtkinter.CTkLabel(master=frame, text = 'Nick Cage Identifier',font = ("Fixedsys", 15, "bold"))
label.pack(pady = 20, padx = 10)

button_test = customtkinter.CTkButton(master = frame, text = 'Select Image', command = Predict)
button_test.pack(pady = 15, padx =10)

button_load = customtkinter.CTkButton(master = frame, text = 'Train Model', command = train_model)
button_load.pack(pady = 15, padx =10)

app.mainloop()