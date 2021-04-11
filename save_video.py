#!/usr/bin/env python3

import datetime
import os
from pathlib import Path

import cv2
import depthai as dai


def capture(high_res=False):
    # Initialize camera pipeline
    pipeline = dai.Pipeline()

    # Setup RGB and side cameras
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_left = pipeline.createMonoCamera()
    cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_right = pipeline.createMonoCamera()
    cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    cams = [cam_rgb, cam_left, cam_right]
    n_cams = len(cams)

    # Setup video encoding for output cameras
    encoders = []  # rgb, left, right, still
    for cam in cams:
        enc = pipeline.createVideoEncoder()
        if cam.getName() != 'MonoCamera':
            cam.video.link(enc.input)
        else:
            cam.out.link(enc.input)
        encoders.append(enc)

    # Set resolution and fps for cameras
    fps = 25
    for cam in cams:
        if cam.getName() != 'MonoCamera':
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P if not high_res
                              else dai.ColorCameraProperties.SensorResolution.THE_4_K)
        else:
            cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P if not high_res
                              else dai.MonoCameraProperties.SensorResolution.THE_720_P)
    for cam, enc in zip(cams, encoders):
        enc.setDefaultProfilePreset(cam.getResolutionSize(), fps, dai.VideoEncoderProperties.Profile.H265_MAIN)

    # Setup video output data stream
    streams = []
    for i in range(n_cams):
        streams.append(pipeline.createXLinkOut())
        streams[i].setStreamName(f"stream{i}")
        encoders[i].bitstream.link(streams[i].input)

    # Setup still image capture
    still_enc = pipeline.createVideoEncoder()
    still_enc.setDefaultProfilePreset(cams[0].getStillSize(), 1, dai.VideoEncoderProperties.Profile.MJPEG)
    cams[0].still.link(still_enc.input)
    still_stream = pipeline.createXLinkOut()
    still_stream.setStreamName(f"still stream")
    still_enc.bitstream.link(still_stream.input)
    encoders.append(still_enc)
    streams.append(still_stream)

    # Setup preview window
    cams[0].setPreviewSize(500, 500)
    streams.append(pipeline.createXLinkOut())
    streams[-1].setStreamName(f"preview stream")
    cams[0].preview.link(streams[-1].input)

    # Create storage folder
    folder = Path(os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    os.makedirs(folder)

    # Connect to camera and start capturing
    with dai.Device(pipeline) as dev:

        # Prepare data queues
        queues = []
        for stream in streams:
            queues.append(dev.getOutputQueue(stream.getStreamName(), maxSize=30, blocking=True))

        # Start the pipeline
        dev.startPipeline()

        # Processing loop
        want_side = False
        still_count = 0
        with open(folder / 'center.h265', 'wb') as color_file, \
             open(folder / 'left.h265', 'wb') as mono1_file, \
             open(folder / 'right.h265', 'wb') as mono2_file:

            while True:
                # Show preview
                preview_frames = queues[-1].tryGetAll()
                for frame in preview_frames:
                    cv2.imshow('Preview', frame.getData().reshape(frame.getWidth(),
                                                                  frame.getHeight(), 3))
                try:
                    # Empty each queue
                    while queues[0].has():
                        queues[0].get().getData().tofile(color_file)
                    if want_side:
                        while queues[1].has():
                            queues[1].get().getData().tofile(mono1_file)
                        while queues[2].has():
                            queues[2].get().getData().tofile(mono2_file)
                except RuntimeError:
                    break

                # Wait for commands
                key = cv2.waitKey(1)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    break
                elif key == ord('s'):
                    want_side = not want_side  # toggle
                elif key == ord('c'):
                    # Capture still images from queue
                    still_frames = queues[-2].tryGetAll()
                    print(len(still_frames))
                    for frame in still_frames:
                        # Decode JPEG and save
                        frame = cv2.imdecode(frame.getData(), cv2.IMREAD_UNCHANGED)
                        cv2.imwrite(folder / f'still{still_count}.jpg', frame)
                        still_count += 1

        # Remove side video files if unused
        no_side = False
        for file in ['left.h265', 'right.h265']:
            if (folder / file).stat().st_size == 0:
                no_side = True
                os.remove(folder / file)

        print("To view, convert the stream file (.h265) into a video file (.mp4), using commands below:")
        cmd = "ffmpeg -framerate {} -i {} -c copy {}"
        stem = Path(folder.stem)
        print(cmd.format(fps, stem / "center.h265", stem / "center.mp4"))
        if not no_side:
            print(cmd.format(fps, stem / "left.h265", stem / "left.mp4"))
            print(cmd.format(fps, stem / "right.h265", stem / "right.mp4"))


if __name__ == '__main__':
    print('Enter the following commands to start recording video:')
    print('v = standard resolution (1080p) video')
    print('h = high resolution (4K) video')
    print('')
    print('Once capturing the following commands can be used on the preview:')
    print('c = capture still image')
    print('s = toggle on and off side camera video recording')
    print('q = stop capturing')

    while True:
        command = input()
        if command == 'v':
            capture()
            break
        elif command == 'h':
            capture(high_res=True)
            break
