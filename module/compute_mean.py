import cv2
import numpy as np

def compute_mean(path):
    cap = cv2.VideoCapture(path)
    for i in range(175):
        ret, image = cap.read()

    imgs = []
    while(True):
        # Capture frame-by-frame
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
        imgs.append(img)
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # cv2.waitKey()

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    mean_img = np.clip(np.mean(imgs, axis=0), 0,255).astype(np.uint8) 
    print('mean_img', mean_img.shape)
    return mean_img