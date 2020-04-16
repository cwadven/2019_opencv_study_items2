import cv2
import numpy as np
import time

min_confidence = 0.5

file_name = "facedetect.mp4"

def detectAndDisplay(frame):
    start_time = time.time()
    img = cv2.resize(frame, None, fx=0.4, fy=0.4) #사이즈를 0.4배율로
    height, width, channels = img.shape #이미지의 배열을 height, width, channels를 가져온다

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0,0,0), True, crop=False)
    #이미지를 blob 형태로 바꿈!

    #blobFromImage(이미지, 각픽셀값의 배율, (가로크기, 세로크기), (모델을 훈련시키는 동안 사용되었던 각 이미지에서 빼야하는 평균), BGR을 RGB로 만들것인가?, 이미지를 자)
    #(가로크기, 세로크기)
    #YOLO는 3가지 크기로 표준화함!
    #(320, 320) --> 빠르지만 정확도가 떨어짐
    #(416, 416) --> 중간 사이
    #(609, 609) --> 느리지만 정확도가 올라감

    net.setInput(blob)
    outs = net.forward(output_layers)
    #blob화 된 이미지를 입력으로 넣은 후, 무엇인지 인지한 값을 outs에 넣는다

    class_ids = [] #인식한 이름을 넣을 곳
    confidences = [] #인식한 정확도를 넣을 곳
    boxes = [] #인식한 위치좌표를 넣을 곳

    for out in outs: #outs 안에 들어있는 것을 가지고 out에 놓는다
        for detection in out: #out안에 있는 것을 detection으로 놓는다
            scores = detection[5:] #아주많은 정보들이 5번째 이후로 있는 것 같다! (고양이 뭐 기타등등 쭈루룩 나열되는것 같다)
            class_id = np.argmax(scores) #그 리스트 안에 가장큰 값의 인덱스를 class_id로 놓는다
            confidence = scores[class_id] #그리고 인덱스를 통해서 그 가장 큰 값을 가져온다
            if confidence > min_confidence: #만약 가져온 값이 0.5보다 클 경우!
                center_x = int(detection[0] * width) #detection[0]에는 비율적인 중앙x값이 있다
                center_y = int(detection[1] * height) #detection[1]에는 비율적인 중앙y값이 있다
                w = int(detection[2] * width) #detection[2]에는 비율적인 가로의 길이의 값이 있다
                h = int(detection[3] * height) #detection[3]에는 비율적인 높이의 길이의 값이 있다

                x = int(center_x - w / 2) #x좌표의 시작를 가져오기 위해서
                y = int(center_y - h / 2) #y좌표의 시작를 가져오기 위해서

                boxes.append([x,y,w,h]) #배열에 전부 넣기
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    #NMSBoxes 그림을 그리다보면 노이즈가 있다 즉 박스안에 박스가 생기는 경우!!!
    #그것을 없애기 위해서 전체 박스에서 가장 괜찮은 박스를 골라서 indexes라는 리스트 안에 넣는다
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)): #전체 박스 숫자만큼 루프를 돌려서
        if i in indexes: #겹치는 박스들에서 거른 indexes만 나오게 한 박스만 보이게 만들기 위해서!
            x, y, w, h = boxes[i] #그것을박스 i녀석의 것을 가져와라
            label = "{} : {:.2f}".format(classes[class_ids[i]], confidences[i]*100) #그 번호를 가져오면 그번호에 맞는 classes를 지정한 이름이 나온다!
            color = colors[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x,y+30), font, 1, color, 1)

    
    end_time = time.time()
    process_time = end_time - start_time
    print("=== frame took {:.3f} seconds".format(process_time))
    cv2.imshow("YOLO Video", img)


#YOLO 모델 로드
#3가지 파일 필요!
#.weights 는 학습된 모델
#yolov3.cfg
#name 파일
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
#weights 랑 yolov3.cfg 모델 적용 (1, 2)
classes = []
#coco.names 이름 가져오기! (3)
with open("coco.names", "r") as f: #파일을 읽음!
    classes = [line.strip() for line in f.readlines()] #f.readlines() 한 줄로 하나씩 읽는다
#classes라는 리스트에 coco.names에 있는 것들이 하나하나 들어간다
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
#yolo의 작동하는 방식
colors = np.random.uniform(0, 255, size=(len(classes), 3))
#80개에 3개 만큼의 채널만큼 해서 0~255 함수를 랜덤하게 준다)

cap = cv2.VideoCapture(file_name)
if not cap.isOpened:
    print("--(!)Error opening video capture")
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print("--(!) No captured frame -- Break!")
        break
    detectAndDisplay(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

