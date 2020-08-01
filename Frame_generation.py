# Importing all necessary libraries
import cv2
import os

# Read the video from specified path
cam = cv2.VideoCapture(r"C:/Users/ANSHIKA SHARMA/PycharmProjects/sih_project/cigratte_in_public_video/CIP_14.mp4")

try:

    # creating a folder named data
    if not os.path.exists('cigratte_in_public_video_data_frames'):
        os.makedirs('cigratte_in_public_video_data_frames')

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 1649


while (True):

    # reading from frame
    ret, frame = cam.read()
    cam.set(cv2.CAP_PROP_FPS, 24)

    if ret:
        # if video is still left continue creating images
        name = './cigratte_in_public_video_data_frames/2_' + str(currentframe) + '.jpg'
        print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
