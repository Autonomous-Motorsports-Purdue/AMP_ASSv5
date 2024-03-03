import time
import pyzed.sl as sl
import rpyc


class OdomTalker(rpyc.Service):

    def __init__(self):
        super().__init__()
        self.odom = Odom()

    def on_connect(self, conn):  # keyword method that allows for diagnostics on connect
        print("Talker connected")
        return "Talked connected"

    def on_disconnect(self, conn):  # keyword like above, but disconnect
        print("Talker disconnected")
        return "Talker disconnected"

    def exposed_get_position(self):  # the exposed keyword at the front allows the object to be accesible.
        return self.odom.get_odom_data()


class Odom:
    def __init__(self):
        # Set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode (default fps: 60)
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  # Use a right-handed Y-up coordinate system
        init_params.coordinate_units = sl.UNIT.FOOT  # Set units in meters

        self.zed = sl.Camera()
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print("Camera Open", status, "Exit program.")
            exit(1)

        tracking_params = sl.PositionalTrackingParameters()
        tracking_params.enable_imu_fusion = False
        status = self.zed.enable_positional_tracking(tracking_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print("Position tracking:", status, "Exit program.")
            exit(1)

        self.runtime = sl.RuntimeParameters()
        self.camera_pose = sl.Pose()
        self.camera_info = self.zed.get_camera_information()
        self.translation = sl.Translation()

    def get_odom_data(self):
        if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
            tracking_state = self.zed.get_position(self.camera_pose, sl.REFERENCE_FRAME.WORLD)
            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                # rotation = camera_pose.get_rotation_vector()
                translation_coords = self.camera_pose.get_translation(self.translation)
                return translation_coords.get()
        else:
            return None


if __name__ == '__main__':
    from rpyc.utils.server import ThreadedServer

    t = ThreadedServer(OdomTalker, port=9001)
    t.start()
