import cv2
import numpy as np
import time

from tensorflow.keras import models
# './layDuLieu/videoData/khanh.mp4'
cam = cv2.VideoCapture(0)
cam.set(3, 640) #width
cam.set(4, 480) #height

models = models.load_model('./model/model9.h5')

# face detection model
face_Detection_Model = './model/res10_300x300_ssd_iter_140000_fp16.caffemodel'

# mô tả kiến trúc
face_Detection_Proto = './model/deploy.prototxt.txt'

# sử dụng OpenCV DNN đọc mô hình nhận diện khuôn mặt
dectector_Model = cv2.dnn.readNetFromCaffe(face_Detection_Proto, face_Detection_Model)

listRrsult = ['Aphat', 'bao', 'dong', 'hieu', 'khanh']

count = 0
while True:
    OK, frame = cam.read()

    if not OK:
        break

    img1 = frame # 1 tấm ảnh

    # chuẩn bị dữ liệu đầu vào cho mô hình nhận diện khuôn mặt
    # scalefactor: tỉ lệ co dãn của hình ảnh,
    # size: kích thước mà mô hình yêu cầu cho đầu vào
    # mean: màu sắc trung bình
    # swapRB: ko hoán đổi kênh màu đỏ v xanh
    # crop: có cắt anảnh hay ko
    imgBlob = cv2.dnn.blobFromImage(img1, 1, (300, 300), (104, 177, 123), swapRB=False, crop=False)

    # thiết lập đầu vào cho mô hình
    dectector_Model.setInput(imgBlob)

    # thực hiện vc nhận diện khuôn mặt
    faces = dectector_Model.forward()

    # lấy width and height (shape[:2]: chỉ lấy 2 dữ liệu đầu tiên)
    h, w = img1.shape[:2]

    for i in range(0, faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        # kiểm tra nếu khuôn mặt có độ tin cậy từ 0.98
        if (confidence >= 0.98):
            # Trích xuất tọa độ
            startX = int(faces[0, 0, i, 3] * w)
            startY = int(faces[0, 0, i, 4] * h)
            endX = int(faces[0, 0, i, 5] * w)
            endY = int(faces[0, 0, i, 6] * h)
            roi = cv2.resize(frame[startY: endY, startX: endX], (200, 200))
            # np.argmax: trả về chỉ số của giá trị lớn nhất
            # time.sleep(0.06)
            result = np.argmax(models.predict(roi.reshape((-1, 200, 200, 3))))
            cv2.rectangle(frame, (startX, startY), (endX, endY), (128, 255, 50), 1)
            cv2.putText(frame, listRrsult[result], (startX, startY - 15), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
