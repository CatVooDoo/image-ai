import tkinter as tk  # импорт библиотеки tkinter
from tkinter import messagebox  # импорт модуля messagebox
from PIL import Image, ImageDraw, ImageOps  # импорт модулей для работы с изображениями
import numpy as np  # импорт библиотеки numpy
import tensorflow as tf  # импорт библиотеки tensorflow
from tensorflow.keras import layers, models  # импорт слоев и моделей keras

class_names = [  # список названий классов
    'самолет', 'автомобиль', 'птица', 'кот', 'олень',
    'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик'
]

def train_model():  # функция для обучения модели
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()  # загрузка данных cifar10
    x_train = x_train / 255.0  # нормализация данных
    x_test = x_test / 255.0  # нормализация данных
    model = models.Sequential()  # создание последовательной модели
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # добавление сверточного слоя
    model.add(layers.MaxPooling2D((2, 2)))  # добавление слоя пулинга
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # добавление сверточного слоя
    model.add(layers.MaxPooling2D((2, 2)))  # добавление слоя пулинга
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # добавление сверточного слоя
    model.add(layers.Flatten())  # добавление слоя выравнивания
    model.add(layers.Dense(64, activation='relu'))  # добавление полносвязного слоя
    model.add(layers.Dense(10))  # добавление выходного слоя
    model.compile(optimizer='adam',  # компиляция модели
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))  # обучение модели
    model.save('cifar10_model.h5')  # сохранение модели
    return model  # возврат обученной модели

try:  # попытка загрузки модели
    model = tf.keras.models.load_model('cifar10_model.h5')  # загрузка модели
except:  # если модель не найдена
    model = train_model()  # обучение модели

class DrawingApp:  # класс для приложения рисования
    def __init__(self, root):  # инициализация приложения
        self.root = root  # установка корневого окна
        self.root.title("Распознавание предметов")  # установка заголовка окна
        self.root.geometry("600x600")  # установка размеров окна
        self.brush_size = 3  # установка размера кисти
        self.color = 'white'  # установка цвета кисти
        self.canvas = tk.Canvas(root, bg='black', width=500, height=400)  # создание холста
        self.canvas.pack(pady=20)  # размещение холста
        self.btn_frame = tk.Frame(root)  # создание фрейма для кнопок
        self.btn_frame.pack()  # размещение фрейма
        self.clear_btn = tk.Button(self.btn_frame, text="Очистить", command=self.clear_canvas)  # кнопка очистки
        self.clear_btn.pack(side=tk.LEFT, padx=10)  # размещение кнопки
        self.predict_btn = tk.Button(self.btn_frame, text="Распознать", command=self.predict_object)  # кнопка распознавания
        self.predict_btn.pack(side=tk.LEFT, padx=10)  # размещение кнопки
        self.result_label = tk.Label(root, text="", font=('Helvetica', 16))  # создание метки для результата
        self.result_label.pack(pady=20)  # размещение метки
        self.image = Image.new("L", (500, 500), 0)  # создание изображения
        self.draw = ImageDraw.Draw(self.image)  # создание объекта для рисования
        self.canvas.bind("<B1-Motion>", self.paint)  # привязка события рисования

    def paint(self, event):  # функция для рисования
        x, y = event.x, event.y  # получение координат
        self.canvas.create_oval(x - self.brush_size,  # рисование овала на холсте
                               y - self.brush_size,
                               x + self.brush_size,
                               y + self.brush_size,
                               fill=self.color, outline=self.color)
        self.draw.ellipse([x - self.brush_size,  # рисование овала на изображении
                          y - self.brush_size,
                          x + self.brush_size,
                          y + self.brush_size],
                          fill=255, outline=255)

    def clear_canvas(self):  # функция очистки холста
        self.canvas.delete("all")  # очистка холста
        self.image = Image.new("L", (300, 300), 0)  # создание нового изображения
        self.draw = ImageDraw.Draw(self.image)  # создание объекта для рисования
        self.result_label.config(text="")  # очистка метки результата

    def predict_object(self):  # функция распознавания объекта
        img = self.image.resize((32, 32))  # изменение размера изображения
        img_rgb = Image.new("RGB", img.size)  # создание rgb изображения
        img_rgb.paste(img)  # вставка изображения
        img_array = np.array(img_rgb) / 255.0  # нормализация изображения
        img_array = np.expand_dims(img_array, axis=0)  # добавление оси
        predictions = model.predict(img_array)  # предсказание модели
        predicted_class = np.argmax(predictions)  # определение класса
        confidence = tf.nn.softmax(predictions[0])[predicted_class]  # вычисление уверенности
        self.result_label.config(  # обновление метки результата
            text=f"Предмет: {class_names[predicted_class]}\nУверенность: {confidence:.2%}"
        )

if __name__ == "__main__":  # запуск приложения
    root = tk.Tk()  # создание корневого окна
    app = DrawingApp(root)  # создание экземпляра приложения
    root.mainloop()  # запуск основного цикла