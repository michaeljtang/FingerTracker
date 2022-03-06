# USAGE
# python scan.py --image images/page.jpg

# import the necessary packages
from dataclasses import dataclass
from typing import Any

from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import cv2
import imutils
import numpy as np

screenCnt = []

def click(event, x, y, flag, image):
    global screenCnt
    if event == cv2.EVENT_LBUTTONDOWN:
        coords = [[x, y]]
        screenCnt.append(coords)

def transform_image(image, output_image="scannedImage.jpg", automatic=True):
    """
    :param image: image frame
    :param paper_dims: dimensions of paper (in pixels) to scale scanned image to
    :param output_image: name of file to write new image to
    :param black_background: if True, will perform page calibration by casting background to black 
    :param automatic: if True, will perform page calibration automatically
    :return: returns transformation matrix
    """
    global screenCnt

    # preserve original image
    orig = image.copy()
    
    if automatic:
        # mask everything from neon green to black: helps distinguish page from background
        # first, turn page to black and background to white
        lower = np.array([30, 50, 50])
        upper = np.array([50, 255, 255])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        # then, flip color so background is black and page is white
        mask[mask > 0] = 1
        new_mask = np.subtract(np.ones(np.shape(mask)), mask)
        new_mask = new_mask.astype('uint8')
        image = cv2.bitwise_and(image, image, mask=new_mask)

        # convert the image to grayscale, blur it, and find edges in image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        # find the contours in the edged image, keeping only the largest ones, and initialize the screen contour 
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then we can assume that we have found the corners
            if len(approx) == 4:
                screenCnt = approx

                # Display tagged corners so that automatic page detection can be manually verified
                red = (0, 0, 255)
                for i in range(4):
                    with_corners = cv2.circle(orig, (approx[i][0][0], approx[i][0][1]), radius=5, color=red, thickness=-1)
                show_image(with_corners)
                break
    # Manual calibration
    else:
        # display image so it can be clicked on
        cv2.namedWindow('CapturedImage', cv2.WINDOW_NORMAL)
        cv2.imshow("CapturedImage", image)

        cv2.setMouseCallback('CapturedImage', click, image)
        while(len(screenCnt) < 4):
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                print('Image cropped at coordinates: {}'.format(screenCnt))
                cv2.destroyAllWindows()
                break
        screenCnt = np.asarray(screenCnt)
    
    # check if we successfully found our screen
    if screenCnt == []:
        return None, None

    # show the contour (outline) of the piece of paper
    print("STEP 2: Find contours of paper")
    # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    # image = cv2.resize(image, (1280,800))
    # cv2.imshow("Outline", image)
    # cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # apply the four point transform to obtain a top-down
    # view of the original image
    M, warped, dims = four_point_transform(orig, screenCnt.reshape(4, 2))
    show_image(warped)
    #show_image(warped)
    #find_markers(warped)
    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # T = threshold_local(warped, 11, offset=10, method="gaussian")
    # warped = (warped > T).astype("uint8") * 255

    # show the original and scanned images
    print("STEP 3: Apply perspective transform")

    # cv2.imwrite(output_image, cv2.resize(warped, paper_dims))
    # cv2.imshow("Scanned", cv2.resize(warped, paper_dims))
    # cv2.waitKey(0)

    print("Done transform initialization")

    return M, dims

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


@dataclass(frozen=True)
class TransformMetadata:
    transformation_matrix: Any
    im_dims: (float, float)
    desired_dimensions: (float, float)


def transform_point(point: (int, int), transform_metadata: TransformMetadata):
    """
    :param point: point in original plane
    :param M: transformation matrix
    :return: prints point that point is transformed to in new plane
    """
    a = np.array([np.array([point], dtype='float32')])
    cur = cv2.perspectiveTransform(a, transform_metadata.transformation_matrix)
    x = cur.flatten()[0] * transform_metadata.desired_dimensions[0] / transform_metadata.im_dims[0]
    y = cur.flatten()[1] * transform_metadata.desired_dimensions[1] / transform_metadata.im_dims[1]
    return x, y


def get_transform_video(video_path, auto_page_calib=True, desired_dimensions=(11.5, 11.0)):
    """
    Takes a video of Braille reading in which the last 5 seconds of the video don't feature
    anything except the page to be read. This function will provide the transformation mapping
    this page, as viewed from an angle to a front-on page view. 

    :param video_path: path to video of Braille reading to be analyzed
    :param auto_page_calib: bool that marks if page calibration should be automatic
    :param desired_dimensions: indicates the actual dimensions of the page in inches
    """
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    m = None

    # auto page calibration
    if auto_page_calib:
        # try auto page calibration with a couple different frames before giving up
        for i in range(75, 25, -5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_length - i)
            ret, frame = cap.read()
            show_image(frame)
            if ret: # ie. if video read was successful
                m, im_dims = transform_image(frame)
            if m is not None:
                break

    # if m is still None after this, then default to manual read
    if m is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_length - 75)
        m, im_dims = transform_image(video_path, automatic=False)

    return TransformMetadata(m, im_dims, desired_dimensions)

# Test Code
if __name__ == '__main__':
    video_path = "./test_images/test_0.mp4"
    get_transform_video(video_path)