import time
import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.util import img_as_ubyte

# Constants #
T = 0.1  # Threshold value
A = 0.05  # Alarm threshold value
alpha = 0.95  # Background update rate


def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)


def capture_from_camera_and_show_images():
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
    cap = cv2.VideoCapture(url)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")
    # Get first image
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame")
        exit()

    # Transform image to gray scale and then to float, so we can do some processing
    background_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background_frame = img_as_float(background_frame)

    # To keep track of frames per second
    start_time = time.time()
    n_frames = 0
    stop = False
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Transform image to gray scale and then to float, so we can do some processing
        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        new_frame_gray = img_as_float(new_frame_gray)

        # Compute difference image
        dif_img = np.abs(new_frame_gray - background_frame)

        # Create a binary image from the difference image using a threshold value T = 0.1
        binary_img = dif_img > T

        # Compute the total number of foreground, F, pixels in the binary image
        F = np.sum(binary_img)

        # Compute the percentage of foreground pixels in the binary image
        H, W = dif_img.shape
        P = F / (H * W) * 100

        # Decide if the percentage of foreground pixels is above a threshold value A = 0.05
        sound_alarm = P > (A * 100)

        # If soundAlarm is true, then display the message “Change detected!” on the new_frame
        if sound_alarm:
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(new_frame, "Change detected!", (100, 350), font, 1, (0, 255, 0), 1)

        # Keep track of frames-per-second (FPS)
        n_frames = n_frames + 1
        elapsed_time = time.time() - start_time
        fps = int(n_frames / elapsed_time)

        # Put the FPS on the new_frame
        str_out = f"fps: {fps}"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, str_out, (100, 100), font, 1, 255, 1)

        # Display the number of changed pixels, F, and the average, minimum and maximum values of the difference image
        str_out = f"F: {F}"
        cv2.putText(new_frame, str_out, (100, 150), font, 1, (120, 120, 0), 1)

        str_out = f"Avg: {np.mean(dif_img):.4f}"
        cv2.putText(new_frame, str_out, (100, 200), font, 1, (120, 120, 0), 1)

        str_out = f"Min: {np.min(dif_img):.4f}"
        cv2.putText(new_frame, str_out, (100, 250), font, 1, (120, 120, 0), 1)

        str_out = f"Max: {np.max(dif_img):.4f}"
        cv2.putText(new_frame, str_out, (100, 300), font, 1, (120, 120, 0), 1)

        # Display the resulting frame
        show_in_moved_window('Input', new_frame, 0, 10)
        show_in_moved_window('Input gray', new_frame_gray, 600, 10)
        show_in_moved_window('Difference image', dif_img, 1200, 10)
        show_in_moved_window('Binary image', img_as_ubyte(binary_img), 0, 400)

        # Updates the background image, background_frame, using:
        # background_frame = alpha * background_frame + (1 - alpha) * new_frame
        background_frame = alpha * background_frame + (1 - alpha) * new_frame_gray

        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_from_camera_and_show_images()
