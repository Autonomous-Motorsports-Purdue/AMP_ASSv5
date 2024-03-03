import time
import pyzed.sl as sl

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720 # Use HD720 video mode (default fps: 60)
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # Use a right-handed Y-up coordinate system
init_params.coordinate_units = sl.UNIT.FOOT# Set units in meters

zed = sl.Camera()
status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("Camera Open", status, "Exit program.")
    exit(1)

tracking_params = sl.PositionalTrackingParameters()
tracking_params.enable_imu_fusion = False
status = zed.enable_positional_tracking(tracking_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("Position tracking:", status, "Exit program.")
    exit(1)

runtime = sl.RuntimeParameters()
camera_pose = sl.Pose()
camera_info = zed.get_camera_information()

py_translation = sl.Translation()
pose_data = sl.Transform()
print("tx, ty, tz")
while True:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        tracking_state = zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
        if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
            # rotation = camera_pose.get_rotation_vector()
            translation = camera_pose.get_translation(py_translation)
            tx = translation.get()[0]
            ty = translation.get()[1]
            tz = translation.get()[2]
    else:
        time.sleep(0.001)