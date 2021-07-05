import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN


class FaceDetector(object):
    """
    Face detector class
    """
    global n
    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def _draw(self, frame, boxes, probs, landmarks):
        """
        Draw landmarks and boxes for each face detected
        """
        try:
            for box, prob, ld in zip(boxes, probs, landmarks):
                # Draw rectangle on frame
                '''
                cv2.rectangle(frame,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255),
                              thickness=2)
                '''


                # Show probability
                '''
                cv2.putText(frame, str(
                    prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                '''

                # Draw landmarks
                '''
                cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)
                '''
        except:
            pass

        return frame

    def run(self):
        """
            Run the FaceDetector and draw landmarks and boxes around detected faces
        """
        cap = cv2.VideoCapture(0)
        n = 0
        #y = 0
        #x = 0
        while True:
            ret, frame = cap.read()
            try:
                # detect face box, probability and landmarks
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                # draw on frame
                y = int(boxes[0][1])
                x = int(boxes[0][0])
                h = int(boxes[0][3])
                w = int(boxes[0][2])
                self._draw(frame, boxes, probs, landmarks)
                print(type(boxes))
                print(boxes)
                print(x,y,w,h)
                #print([boxes[0][0]], [boxes[0][1]])
                cv2.imwrite(f'./face_images/2_face{n}.png',frame[y:h, x:w])
                n+=1
                if n > 100:
                    break

            except:
                pass

            # Show the frame
            cv2.imshow('Face Detection', frame)
            #print(type(frame))


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Run the app
mtcnn = MTCNN()
fcd = FaceDetector(mtcnn)
fcd.run()