#Face Dtector by AstroMh

import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils
mpFace = mp.solutions.face_detection

with mpFace.FaceDetection(model_selection=0, min_detection_confidence=0.5) as FaceDetection:
    while True:
        ret, img = cap.read()
        count = 0
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = FaceDetection.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(results.detections)
        
        #Drawing Dtections on screen
        if results.detections:
            for detection in results.detections:
                mpDraw.draw_detection(image, detection)
        else:
            continue
        
        #Counting Detctions from screen
        for count, detection in enumerate(results.detections):
          count += 1
        image = cv2.putText(image, '('+ str(count) + ')'+ ' Faces Detected!', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (24, 70, 222), 1, cv2.LINE_AA)
        
        cv2.imshow('Computer Vision', image) 
        if cv2.waitKey(10) & 0xff==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()