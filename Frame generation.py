import cv2
vidcap = cv2.VideoCapture('C:/Users/risha/Downloads/Video/y2mate.com - 1 minute motivational video._bKxL536zqro_360p.mp4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("image"+str(count)+".png", image)     # save frame as png file
    return hasFrames
sec = 0
frameRate = 1 #//it will capture image in each second
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)