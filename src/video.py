import time
import datetime
import cv2 as cv


class Video:
    def __init__(self, width=1000, height=500, videoInput=0, bufferSize=0):

        self.videoInput = videoInput
        self.width = width
        self.height = height
        self.bufferSize = bufferSize
        self.faces = []
        self.saved = 0

        self.capture = cv.VideoCapture(self.videoInput, cv.CAP_DSHOW)
        self.time = 0

        if not self.capture.isOpened():
            raise ValueError(f"Could not open video input.")

        self.capture.set(cv.CAP_PROP_BUFFERSIZE, self.bufferSize)
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)

        # load pre trained haar cascade classifier
        self.faceClassifier = cv.CascadeClassifier(
            'haarcascade_frontalface_default.xml'
        )

    def startStream(self, g):
        startTime = time.time()

        while True and g:
            key = cv.waitKey(1)
            ret, frame = self.capture.read()

            if not ret:
                print("unable to retrieve frame")
                break

            now = str(datetime.datetime.now())
            currTime = self.parseTime(now)

            # display date and current time in upper left corner 
            # (10, 2) and (205, 20) ordered pair on image
            # colour of rectangle white (255, 255, 255)
            # -1: fill rectangle
            cv.rectangle(frame, (10, 2), (205, 20), (255, 255, 255), -1)
            cv.putText(frame, currTime, (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            
            # steps for face detection
            grayFrame = self.toGrayScale(frame)
            self.faces = self.getFaces(grayFrame)

            if len(self.faces) > 0:
                # perform gaussian blur on each face
                for i in range(len(self.faces)):
                    frame = self.blurFace(frame, i)
            

            cv.imshow("stream", frame)

            if key == ord('q'):
                finalTime = time.time()
                self.time = finalTime-startTime
                break        
            # switch to unblurred faces
            if key == ord('f'):
                self.startStream(g=False)
            # save current frame
            if key == ord('s'):
                cv.imwrite(f"imgs\saved_img{self.saved}.jpg", frame)
                self.saved += 1
                print("image saved")
        

        # -------------------------------------------------------------------------------------------------------------

        while True and not g:
            key = cv.waitKey(1)
            ret, frame = self.capture.read()

            if not ret:
                print("unable to retrieve frame")
                break

            now = str(datetime.datetime.now())
            currTime = self.parseTime(now)

            # display date and current time in upper left corner 
            # (10, 2) and (205, 20) ordered pair on image
            # colour of rectangle white (255, 255, 255)
            # -1: fill rectangle
            cv.rectangle(frame, (10, 2), (205, 20), (255, 255, 255), -1)
            cv.putText(frame, currTime, (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            cv.imshow("stream", frame)


            if key == ord('q'):
                finalTime = time.time()
                self.time = finalTime-startTime
                break

            # switch to blurred facqes
            if key == ord('g'):
                self.startStream(True)
            
            if key == ord('s'):
                cv.imwrite(f"imgs\saved_img{self.saved}.jpg", frame)
                self.saved += 1
                print("image saved")

    def getFaces(self, grayScaleFrame):
        # smaller scale factors=more accurate detection at cost of computation time
        # larger scale factors=reduce comp time but may miss faces
        # higher neighbour value means more strict detector, a lower neighbour value
        # makes detector more sensitive, potentially detecting more faces, including false positives
        return self.faceClassifier.detectMultiScale(grayScaleFrame, scaleFactor=1.05, minNeighbors=8)

    def toGrayScale(self, frame):
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    def releaseVideo(self):
        cv.destroyAllWindows()
        self.capture.release()
        print("Released video")
        return self.time

    def parseTime(self, time):
        return time[0:19]

    # either give this function the face rectangle and blur the whole thing
    # or give it the frame and the face boundary and blur that specific
    # region ?
    def blurFace(self, frame, i):

        # get ordered pair of points for rectangle containing face
        # x1, x2 are the first two elements
        # to find x2 and y2, we add the width and height respectively    
        x1 = self.faces[i][0]
        y1 = self.faces[i][1]
        x2 = x1 + self.faces[i][2]
        y2 = y1 + self.faces[i][3]

        # only blur the image within the region provided
        # since we want a heavy blur, we give a larger kernel 
        # which will be including values of further away pixels
        blurredROI = cv.GaussianBlur(frame[y1:y2, x1:x2], (67, 67), 0)
        frame[y1:y2, x1:x2] = blurredROI

        return frame
    
    #def writeFace(self)

    """
    def findFace(self, frame, i)
        # get ordered pair of points for rectangle containing face
        # x1, x2 are the first two elements
        # to find x2 and y2, we add the width and height respectively    
        x1 = self.faces[i][0]
        y1 = self.faces[i][1]
        x2 = x1 + self.faces[i][2]
        y2 = y1 + self.faces[i][3]
    """

