import cv2
from random import randint

import scan
import BraillePage
import os


def show_image(image):
    """
    Displays image using cv2, but rescaled to fit on standard screen.
    Primarily intended for viewing images while debugging.
    
    :param image: image to be displayed
    """
    resized = cv2.resize(image, (1280, 800))
    cv2.imshow('Image', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class VideoTracker:
    """Video Tracker Class"""
    trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

    def __init__(self, video_path, page_path=None, tracker_type="CSRT", auto_finger_calib=False, auto_page_calib=True, output_path='./test_output/output.mp4',
                 show_frame=False):
        """
        init video tracker
        :param video_path: Path of input video
        :param page_path: Path of input Braille page being read
        :param tracker_type: type of OpenCV tracker to use, CSRT seems to work the best so far and is default
        :param auto_finger_calib: if True, will use pre-defined
        :param auto_page_calib: If True, will use automatic bounding boxes instead of manual
        :param output_path: Path of output video and txt, will create if does not exist
        :param show_frame: If true, tracker displays the frame at each iteration
        """
        # Create a video capture object to read videos
        self.cap = cv2.VideoCapture(video_path)

        # load in page of Braille
        self.braille_page = BraillePage.BraillePage(page_path) 

        # video info
        self.vid_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output info
        self.output_path = output_path

        # Get page transform
        self.transformation_metadata = scan.get_transform_video(video_path, (11.5, 11))

        # Calibration of boxes
        if auto_finger_calib:
            # use predefined bounding boxes
            bboxes, frame, colors = self.automatic_calibration()
        else:
            # manually draw bounding boxes
            bboxes, frame, colors = self.manual_calibration()

        # initialize multitracker object based on bounding boxes and selected tracker type
        multi_tracker = self.init_multitracker(bboxes, tracker_type, frame)

        # video saving format
        output_format = cv2.VideoWriter_fourcc(*'mp4v')

        # open and set properties
        video_out = cv2.VideoWriter()
        video_out.open(self.output_path, output_format, self.fps, (self.vid_width, self.vid_height), True)

        # run tracker and save video
        x_centers, y_centers, letters = self.process_tracker(self.cap, multi_tracker, colors, video_out, show_frame)
        
        ### FOR TESTING
        # print(x_centers, y_centers, letters)

    def init_multitracker(self, bboxes, tracker_type, frame):
        """
        Init an Opencv tracker instance, given a set of bounding boxes (e.g. one for each finger), and type
        :param bboxes: Bounding boxes
        :param tracker_type: E.g. CSRT
        :param frame: First frame, used for init
        :return: multitracker instance
        """
        # Create MultiTracker object
        multi_tracker = cv2.MultiTracker_create()

        # Initialize MultiTracker
        for bbox in bboxes:
            multi_tracker.add(self.create_tracker_by_name(tracker_type), frame, bbox)
        return multi_tracker

    def create_tracker_by_name(self, tracker_type):
        """
        Given input string, init the correct tracker
        :param tracker_type: e.g. 'CSRT'
        :return:
        """
        # Create a tracker based on tracker name
        if tracker_type == self.trackerTypes[0]:
            tracker = cv2.TrackerBoosting_create()
        elif tracker_type == self.trackerTypes[1]:
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == self.trackerTypes[2]:
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == self.trackerTypes[3]:
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == self.trackerTypes[4]:
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_type == self.trackerTypes[5]:
            tracker = cv2.TrackerGOTURN_create()
        elif tracker_type == self.trackerTypes[6]:
            tracker = cv2.TrackerMOSSE_create()
        elif tracker_type == self.trackerTypes[7]:
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = None
            print('Incorrect tracker name')
            print('Available trackers are:')
            for t in self.trackerTypes:
                print(t)

        return tracker

    def read_frame(self, second):
        """
        Reads frame at given time stamp of video
        :param second: the time stamp of the video to read, in seconds
        :return: the frame at the input time stamp
        """
        # Read frame
        if second == 0:
            success, frame = self.cap.read()
        else:
            frame_count = int(second * self.fps)
            for i in range(frame_count):
                self.cap.grab()
            success, frame = self.cap.retrieve()

        try:
            frame = cv2.resize(frame, (1920, 1080))
        except cv2.error as err:
            print("Failed to read video")
            raise err

        self.vid_width = 1920
        self.vid_height = 1080

        # quit if unable to read the video file
        if not success:
            print('Failed to read video')
            raise Exception("Failed to read video")

        return frame

    def manual_calibration(self):
        """
        Manually draw bounding boxes
        :return: bounding boxes, first frame, color of each box
        """
        # TODO - should just be 0, right now its 5 just cause of poor test video
        frame = self.read_frame(5)

        # Select boxes
        bboxes = []
        colors = []

        # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
        # So we will call this function in a loop till we are done selecting all objects
        while True:
            # draw bounding boxes over objects
            # selectROI's default behaviour is to draw box starting from the center
            # when fromCenter is set to false, you can draw box starting from top left corner
            cv2.namedWindow('MultiTracker', 2)
            #cv2.setWindowProperty("MultiTracker",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

            bbox = cv2.selectROI('MultiTracker', frame)
            bboxes.append(bbox)
            colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
            print("Press q to quit selecting boxes and start tracking")
            print("Press any other key to select next object")
            k = cv2.waitKey(0) & 0xFF
            if k == 113:  # q is pressed
                break

        print('Selected bounding boxes {}'.format(bboxes))

        ################################
        return bboxes, frame, colors

    def automatic_calibration(self):
        """
        Draw bounding boxes from predefined coordinates
        :return: bounding boxes, first frame, color of each box
        """
        defined_calibration_pts = [(371, 887, 77, 68),
                                   (571, 999, 99, 46),
                                   (692, 991, 111, 56),
                                   (801, 983, 95, 64),
                                   (998, 991, 93, 56),
                                   (1100, 981, 98, 66),
                                   (1248, 983, 101, 63),
                                   (1359, 881, 94, 60)]
        # frame = self.read_frame(0)

        # Select boxes
        bboxes = []
        colors = []

        for this_box in defined_calibration_pts:
            bboxes.append(this_box)
            this_color = randint(0, 255), randint(0, 255), randint(0, 255)
            colors.append(this_color)

        return bboxes, frame, colors

    def generate_output_file(self, x_centers, y_centers, letters):
        """
        Generates tab delimited output file
        :param x_centers: list of center of each box, x coord
        :param y_centers: list of center of each box, y coord
        :param letters: list of the letter on the Braille page corresponding to the input x and y coords
        :return:
        """
        with open('BrailleOutput.txt', 'w+') as outfile:
            outfile.write('Frame\tX1'
                          '\tY1'
                          '\tX2'
                          '\tY2'
                          '\tX3'
                          '\tY3'
                          '\tX4'
                          '\tY4'
                          '\tX5'
                          '\tY5'
                          '\tX6'
                          '\tY6'
                          '\tX7'
                          '\tY7'
                          '\tX8'
                          '\tY8'
                          '\tL1'
                          '\tL2'
                          '\tL3'
                          '\tL4'
                          '\tL5'
                          '\tL6'
                          '\tL7'
                          '\tL8\n')

            for i in range(len(x_centers)):
                # formatting a single line containing coords X1, Y1 through X8, Y8
                frame_str = str(i) + '\t' + "\t".join(["{0}\t{1}".format(x, y) for x, y in zip(x_centers[i], y_centers[i])])

                # adding the associated letters to frame_str
                letter_str = '\t' + "\t".join(["{0}".format(letter) for letter in letters[i]])
                frame_str += letter_str

                # for box in range(8):
                #     if box != 7:
                #         row_data += str(x_centers[i][box]) + '\t' + str(y_centers[i][box]) + '\t'
                #     else:
                #         row_data += str(x_centers[8][box]) + '\t' + str(y_centers[8][box])
                outfile.write(frame_str + '\n')
        return x_centers, y_centers, letters

    def process_tracker(self, cap, multi_tracker, colors, video_out, show_frame=False):
        """
        Given captured video & tracker object, track objects and output video + coordinates
        :param cap: OpenCV Cap object representing video stream
        :param multi_tracker: OpenCV tracker object
        :param colors: Colors of each bounding box
        :param video_out: Output video path
        :param show_frame: If True, program updates the frame during processing
        :return:
        """
        # Initialize Coordinate List and Associated Letter List
        x_centers = []
        y_centers = []
        letters = []
        frame_num = 0

        # Process video and track objects
        while cap.isOpened():
            #print("Processing frame: {0}".format(frame_num))

            success, frame = cap.read()
            if not success:
                break

            try:
                if not success:
                    break
                frame = cv2.resize(frame, (1920, 1080))
            except Exception("Frame read failure"):
                break

            # get updated location of objects in subsequent frames
            success, boxes = multi_tracker.update(frame)

            x_centers_per_frame = [0] * 8
            y_centers_per_frame = [0] * 8
            letters_per_frame = [0] * 8

            # draw tracked objects
            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

                x_center_pixel = boxes[i][0] + boxes[i][2] / 2
                y_center_pixel = boxes[i][1] + boxes[i][3] / 2

                x_centers_per_frame[i], y_centers_per_frame[i] = scan.transform_point((x_center_pixel, y_center_pixel), self.transformation_metadata)
                letters_per_frame[i] = self.braille_page.position2Char(x_centers_per_frame[i], y_centers_per_frame[i])
                #x_centers_per_frame[i] = x_center_pixel
                #y_centers_per_frame[i] = y_center_pixel

            # add coordinates from this frame to overall coordinate list
            x_centers.append(x_centers_per_frame)
            y_centers.append(y_centers_per_frame)
            letters.append(letters_per_frame)
            frame_num += 1
            #print(x_centers, y_centers)

            # show frame
            if show_frame:
                cv2.imshow('MultiTracker', frame)
                video_out.write(frame)

            # quit on ESC button
            if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
                break

        x_centers, y_centers, letters = self.generate_output_file(x_centers, y_centers, letters)
        return x_centers, y_centers, letters


if __name__ == '__main__':
    #video_path = "./videotest2/C0013.mp4"
    video_path = './test_images/test_0.mp4'
    page_path = './braille_files/B_2019_project_FingerTracker.brf'
    tracker = VideoTracker(video_path, page_path=page_path, auto_finger_calib=False, auto_page_calib=False, show_frame=True)
