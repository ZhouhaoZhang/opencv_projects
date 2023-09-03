import pyzed.sl as sl
import time

# Create a ZED camera object
zed = sl.Camera()

# Set camera settings
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.VGA
init_params.camera_fps = 100
init_params.depth_mode = sl.DEPTH_MODE.NONE

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(-1)

# Create a video recorder object
recorder = sl.Recorder("output.avi")

# Start recording
recorder.record(zed)

# Wait for the recording to finish
while recorder.is_recording():
    time.sleep(1)

# Stop recording
recorder.stop()

# Close the camera
zed.close()
