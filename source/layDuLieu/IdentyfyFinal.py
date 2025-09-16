import cv2
import time
import numpy as np


camera = cv2.VideoCapture('./videoData/linh.mp4')
camera.set(3, 640) #width
camera.set(4, 480) #height
# face detection model
face_Detection_Model = './model/res10_300x300_ssd_iter_140000_fp16.caffemodel'

# mô tả kiến trúc
face_Detection_Proto = './model/deploy.prototxt.txt'

# sử dụng OpenCV DNN đọc mô hình nhận diện khuôn mặt
dectector_Model = cv2.dnn.readNetFromCaffe(face_Detection_Proto, face_Detection_Model)


"""
Tóm lại function này là: đầu tiên đọc ảnh, copy ảnh (ko xử lí trên ảnh gốc), nhận
diện face lọc ra face có confidence max sau đó crop lấy vùng face đó trên tấm ảnh copy
"""

count = 0
while True:
    ret, frame = camera.read()
    # nếu có nhìu hình ảnh thì phải bỏ đoạn dưới vào for
    #chuẩn bị dữ liệu đầu vào

    if not ret:
        break

        # trích xuất đặc trưng cho 1 ảnh
        # img1 = cv2.imread(imgInput)
    img1 = frame  # (1 tấm ảnh)

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
    # duyệt từng khuôn mặt đã được phát hiện
    time.sleep(0.08)
    for i in range(0, faces.shape[2]):
        randomGamma = np.random.randint(-120, -100)
        confidence = faces[0, 0, i, 2]
        # kiểm tra nếu khuôn mặt có độ tin cậy từ 0.98
        if (confidence >= 0.98):
            # Trích xuất tọa độ
            startX = int(faces[0, 0, i, 3] * w)
            startY = int(faces[0, 0, i, 4] * h)
            endX = int(faces[0, 0, i, 5] * w)
            endY = int(faces[0, 0, i, 6] * h)

            roi = cv2.resize(frame[startY: endY, startX: endX], (200, 200))

            roiGray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roiContrast = cv2.addWeighted(roiGray, 1, np.zeros(roiGray.shape, roiGray.dtype), 0, randomGamma)

            print(roiContrast.shape)

            count += 1
            print(count)

            # vẽ hcn xung quanh khuôn mặt đã đc phát hiện
    cv2.imshow('keq', img1)

    if (cv2.waitKey(1) == ord('q') or count >= 1000):
        break

camera.release()
cv2.destroyAllWindows()