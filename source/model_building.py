import numpy as np
import os
from PIL import Image
import cv2

import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models

# LƯU Ý CỰC KÌ QUAN TRỌNG: lượng dữ liệu giữa các đối tượng phải tương đối ngang nhau
# nếu 1 đối tượng nào mà có lượng data train lớn hơn hẳn những đối tượng còn lại
# thì rất có thể Ai chỉ có thể nhận dạng được đối tượng đó

# vì theo model của tensorflow thì Xtrain là ma trận đa chiều chỉ chứa dữ liệu ảnh
# mà hiện tại Xtrain là list nên là giờ phải xử lí theo cái form của model
# tensorflow
TRAIN_DATA = 'layDuLieu/dataGray'

# Xtrain = [(matranhinh1, ohe1), (matranhinh2, ohe2), ..... (matranhinh986, ohe986)]
# Xtrain[0][0], Xtrain[0][1]
# là những x sao cho x nằm trong enumerate của Xtrain
# i chính là số lần lặp (len của list Xtrain) x chính là từng tuple trong Xtrain
# vậy thì x[0] sẽ là ma trận hình của từng tuple
# Xtrain = [x[0] for i, x in enumerate(Xtrain)]
# ví dụ về enumerate có thể đưa xuống sau khi Xtrain nhận đc giữ liệu
#   y2 = (x[0] for i, x in enumerate(Xtrain))
#   for index in y2:
#     print(index)

# Xtrain chưa ma trận hình của 3 thư mục trong trainData
Xtrain = []
# Ytrain chứa label đi kèm với từng ma trận hình
Ytrain = []

# để chuyển label thành one hot coding
dict = {
    'Aphat': [1, 0, 0, 0, 0], 'bao': [0, 1, 0, 0, 0], 'dong': [0, 0, 1, 0, 0],
    'hieu': [0, 0, 0, 1, 0], 'khanh': [0, 0, 0, 0, 1]
}

def getData(dirData, listData):
    for whatever in os.listdir(dirData):
        # lập tức viết đường dẫn đi đến whatever
        whatever_path = os.path.join(dirData, whatever)
        # tạo 1 list cho 1 thư mục
        list_fileName_path = []

        for fileName in os.listdir(whatever_path):
            fileName_path = os.path.join(whatever_path, fileName)
            # dùng split để cắt 1 chuỗi dựa vào \ nó sẽ ra 1 list
            # và label nằm ở trí trí thứ 1 của list đó

            label = fileName_path.split('\\')[1]

            # dùng Image mở 1 ảnh từ đường link thì nó sẽ là 1 object
            # sau đó dùng np.array để chuyển nó sang ma trận ảnh
            img = np.expand_dims(np.array(Image.open(fileName_path)), axis=-1)
            # img = np.repeat(img, 3, -1)
            img = np.repeat(img, 3, axis=-1)
            # appnend 1 tuble gồm ma trận hình cũng vs label của nó
            list_fileName_path.append((img, dict[label]))


        # nhưng giờ nếu dùng extend thì nó sẽ phá ngoặc vuông
        # của từng list đi để nó trong hết lại thành 1 Xtrain dài
        listData.extend(list_fileName_path)
    return listData


Xtrain = getData(TRAIN_DATA, Xtrain)
# print(Xtrain.shape)
# cv2.imshow('{0}'.format(Xtrain[2220][1]), Xtrain[2220][0])
# cv2.waitKey(0)



model_training_first = models.Sequential([
    layers.Conv2D(16, (3, 3), input_shape=(200,200, 3),activation='relu'),
    layers.MaxPool2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),


    layers.Flatten(),
    layers.Dense(1000, activation='relu'),
    layers.Dense(500, activation='relu'),
    layers.Dense(250, activation='relu'),
    # giảm xuống lớp dự đoán
    layers.Dense(5, activation='softmax'),
 ])

# model_training_first.summary()
model_training_first.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

# chuyển đổi toàn bộ list sang numpy array
model_training_first.fit(
                         np.array([x[0] for i, x in enumerate(Xtrain)]),
                         np.array([y[1] for i, y in enumerate(Xtrain)]),
                         epochs=10,
                        )
model_training_first.save('model9.h5')


