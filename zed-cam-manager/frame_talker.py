import rpyc
import pyzed.sl as sl
import numpy as np
import pickle

class Talker(rpyc.Service):

    def __init__(self):
        # Create a ZED camera object
        self.zed = sl.Camera()

        # Set configuration parameters
        input_type = sl.InputType()
        init = sl.InitParameters(input_t=input_type)
        init.camera_resolution = sl.RESOLUTION.HD1080
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.coordinate_units = sl.UNIT.FOOT

        # Open the camera
        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            self.zed.close()
            exit(1)

        self.runtime = sl.RuntimeParameters()

        print("ZED camera connected")

    def on_connect(self, conn):
        print("Talker connected")
        return "Talker connected"
    def on_disconnect(self, conn):
        print("Talker disconnected")
        return "Talker disconnected"

    def exposed_get_rgb_frame(self):

        # Capture a new image
        if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns ERROR_CODE.SUCCESS
            image = sl.Mat()
            self.zed.retrieve_image(image, sl.VIEW.LEFT)
            # return np.copy(image.get_data()).tobytes()
            return pickle.dumps(np.copy(image.get_data()))
        else:
            return None
    def exposed_get_depth_frame(self):
        if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns ERROR_CODE.SUCCESS
            image = sl.Mat()
            self.zed.retrieve_image(image, sl.VIEW.DEPTH)
            # return np.copy(image.get_data()).tobytes()
            return pickle.dumps(np.copy(image.get_data()))
        else:
            return None

    def exposed_get_camera_params(self):
        return self.zed.get_camera_information()

def main():
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(Talker, port=9002)
    t.start()

if __name__ == "__main__":
    main()
