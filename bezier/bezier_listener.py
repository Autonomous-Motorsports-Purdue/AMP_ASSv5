import rpyc
import time
import cv2
import numpy as np
import pickle


rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

"""
# need to get the depth and camera parameters from the Zed
def convert_to_real(point):
    u = point[0] # X coordinate from image
    v = point[1] # Y coordinate from image
    # Z = depth value of the pixel from the depth map
    # How do I get the depth value from the zed?

    # c_x = Optical center along x axis, defined in pixels
    # c_y = Optical center along y axis, defined in pixels
    # f_x = Focal length in pixels along x axis. 
    # f_y = Focal length in pixels along y axis. 
    X = ((u - c_x) * Z) / (f_x)
    Y = ((v - c_y) * Z) / (f_y)
    return Z, X, Y
"""
if __name__ == "__main__":
    c = rpyc.connect("localhost", 9001)
    print("Listener Connected")
    while True:
        # img, curve = c.root.get_bezier()
        pickle_img, control_points = c.root.get_bezier()
        if pickle_img is not None:
            img = pickle.loads(pickle_img)
            cv2.imshow("img", img)
            
            # binary_file = open("blistener.txt", "wb")
            # binary_file.write(img_bytes)
            # binary_file.close()

            # np_arr = np.frombuffer(img_bytes, np.uint8)
            # np_copy = np_arr.copy()
            # img = cv2.imdecode(np_copy, cv2.IMREAD_COLOR)
            # cv2.imshow("img", img)

   
        if control_points is not None:
            print(control_points)

        """
        if img2 is not None:
            print("Got image")
            # do something with the image
            print(img)
            cv2.imshow('Image', img2)
        """
        # if curve is not None:
            # print("Got curve")
            # do something with the curve
            # print(curve)
        time.sleep(1)
