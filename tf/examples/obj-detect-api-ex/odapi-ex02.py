# 동영상에서 객체 검출
import cv2
from usbcam import UsbCam
from objdetect import ObjDetectApi
PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
api = ObjDetectApi(MODEL_NAME, PATH_TO_LABELS)

def intrusion_detection(output_dict):
    persons=[]
    for ix, obj_ix in enumerate(output_dict['detection_classes']):
        if obj_ix == 1 and output_dict['detection_scores'][ix]>=0.5:
            persons.append(ix)

        return len(persons)

def detect(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output_dict = api.inference_image(frame_rgb)    # 객체판별 결과를 output_dict가 가지고 있음


    # 이 부분에 있으면 안되는 객체 발견하면 침입발생 !! 침입인지 아닌지 판단
    # 판별할 수 있는 객체의 가지수는 (data/mscoco_label_map.pbtxt)에서 확인, ppt에서 확인
    # 침입발생경우는 영상 속에 사람이 있으면 경고를 줘야 함
    if intrusion_detection(output_dict):
        print("침입 발생")  # 인식하면 계속 침입발생 하는게 아니라 최초 1번 감지 시 1번만 알림
    
        # 몇분동안 인식이 없으면 초기화시켜줘야함
        # 레코딩 시작
        # 카톡으로 알림 전송 등 후속 처리 


    labeled_image = api.visualize(frame_rgb, output_dict)
    labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', labeled_image)
    key = cv2.waitKey(1)
    if key == 27: 
        return False
    else:
        return True
cam= UsbCam()
cam.run(detect)