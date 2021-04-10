#!/usr/bin/env python3

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
    encoders = []
    for cam in cams:
        enc = pipeline.createVideoEncoder()
        if cam.getName() == 'MonoCamera':
            cam.out.link(enc.input)
        else:
            cam.video.link(enc.input)
        encoders.append(enc)

    # Set resolution for cameras
    if high_res:
        cams[0].setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        encoders[0].setDefaultProfilePreset(3840, 2160, 25, dai.VideoEncoderProperties.Profile.H265_MAIN)
        for cam, enc in zip(cams[1:], encoders[1:]):
            cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            enc.setDefaultProfilePreset(1280, 720, 25, dai.VideoEncoderProperties.Profile.H265_MAIN)
    else:
        cams[0].setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        encoders[0].setDefaultProfilePreset(1920, 1080, 25, dai.VideoEncoderProperties.Profile.H265_MAIN)
        for cam, enc in zip(cams[1:], encoders[1:]):
            cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            enc.setDefaultProfilePreset(640, 400, 25, dai.VideoEncoderProperties.Profile.H264_MAIN)

    # Setup video output data stream
    streams = []
    for i in range(n_cams):
        streams.append(pipeline.createXLinkOut())
        streams[i].setStreamName(f"stream{i}")
        encoders[i].bitstream.link(streams[i].input)

    # Connect to camera and start capturing
    with dai.Device(pipeline) as dev:

        # Prepare data queues
        queues = []
        for i in range(n_cams):
            queues.append(dev.getOutputQueue(f"stream{i}", maxSize=30, blocking=True))

        # Start the pipeline
        dev.startPipeline()

        # Processing loop
        with open('color.h265', 'wb') as fileColorH265, open('mono1.h264', 'wb') as fileMono1H264, \
                open('mono2.h264', 'wb') as fileMono2H264:
            print("Press Ctrl+C to stop encoding...")
            while True:
                try:
                    # Empty each queue
                    while queues[0].has():
                        queues[0].get().getData().tofile(fileColorH265)
                    while queues[1].has():
                        queues[1].get().getData().tofile(fileMono1H264)
                    while queues[2].has():
                        queues[2].get().getData().tofile(fileMono2H264)

                except KeyboardInterrupt:
                    break

        print("To view the encoded data, convert the stream file (.h264/.h265) into a video file (.mp4), using commands below:")
        cmd = "ffmpeg -framerate 25 -i {} -c copy {}"
        print(cmd.format("mono1.h264", "mono1.mp4"))
        print(cmd.format("mono2.h264", "mono2.mp4"))
        print(cmd.format("color.h265", "color.mp4"))


if __name__ == '__main__':
    capture()
