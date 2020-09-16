
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from compute_mean import compute_mean

def smooth(x,window_len=30,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError

    if x.size < window_len:
        raise ValueError


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

 
path ='result.avi'
'''
mean_img = compute_mean(path)
plt.imshow(mean_img, cmap='gray')
plt.show()
cap = cv2.VideoCapture(path)
for i in range(175):
    ret, image = cap.read()
'''
cap = cv2.VideoCapture(path)
while(True):
    # Capture frame-by-frame
    ret, image_bgr = cap.read()
    if not ret:
        break
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
    thresh = binary/255
    thresh = 1 - thresh
    nonzero = np.count_nonzero(thresh, axis=0)

    nonzero = smooth(nonzero)
    

    # minute, hour = nonzero[argrelextrema(nonzero, np.greater)[0]]
    pos_local_maximum = argrelextrema(nonzero, np.greater)

    value_local_maximum = nonzero[pos_local_maximum[0]]
    top2_idx_local_maximum = value_local_maximum.argsort()[-2:][::-1] 
    top2_pos_local_maximum = pos_local_maximum[0][top2_idx_local_maximum] # minute, hour, e.g. [230 973]
    minute, hour = top2_pos_local_maximum
    cv2.line(image_bgr, (minute,0), (minute, 200), (255,0,0), 5) # blue
    cv2.line(image_bgr, (hour,0), (hour, 200), (0,0,255), 5) # red
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

    plt.figure(1)
    plt.plot(nonzero)
    plt.figure(2)
    plt.imshow(np.hstack([image_bgr[:,:,::-1], np.stack([binary, binary, binary], -1)]) )
    

    plt.show()


    # cv2.imshow('frame',image_bgr)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    # cv2.waitKey()

# When everything done, release the capture
