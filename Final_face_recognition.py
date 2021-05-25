import cv2
from utility import face_detector, landmark_detector

vid = cv2.VideoCapture(0)
while True:

    ret, img = vid.read()
        
    faces, img = face_detector(img)
    for f in faces:
        mark, mask  = landmark_detector(img[f[1]:f[3], f[0]:f[2]])
        for m in range(5):
            cv2.circle(img, (int(mark[0 + m] + f[0]), int(mark[4 + m] + f[1])), 2, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(img, mask, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        
    cv2.imshow("face", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release()
cv2.destroyAllWindows()