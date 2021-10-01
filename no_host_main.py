#!/usr/bin/env python3

"""
Submatter head movement tracker
Matt MacDonald, Subhash Talluri 2021

Main camera controller
1. Starts a video stream, finds and labels face, estimates head pose angles, calculates velocities
2. Logs all data in json for upload
2. Shows annotated low res preview stream optionally
3. Stores high res video for subsequent analysis on the cloud

This no host version deploys the intermediate processing steps on the camera as much as possible
For simplicity this version does not support advanced face identification

Pre-Trained Models
__________________
Face detection:
OpenVINO model zoo - https://docs.openvinotoolkit.org/latest/omz_models_group_intel.html
face-detection-retail-0005

Head pose estimation:
Hopenet-lite - https://github.com/OverEuro/deep-head-pose-lite
"""


import os
import glob
import shutil
import time
import datetime
from pathlib import Path

import argparse
import signal
import warnings

import queue
import threading
import json

import cv2
import numpy as np
import depthai as dai

from math import cos, sin
from scipy.special import softmax
from imutils.video import FPS

# TODO add velocities support
# TODO implement https://docs.luxonis.com/projects/api/en/latest/samples/spatial_location_calculator/


# Set arguments for running from CLI
parser = argparse.ArgumentParser()
parser.add_argument('-file', '--file', type=str,
                    help="Path to video file to be used for inference instead of live camera")
parser.add_argument('-mins', '--minutes', type=int, default=2,
                    help="Session time out in minutes (default 2 mins)")
parser.add_argument('-nop', '--no-preview', action="store_true",
                    help="Prevent display of an annotated video real time in a preview window")
parser.add_argument('-nov', '--no-video', action="store_true",
                    help="Prevent saving unannotated source video file")
parser.add_argument('-deb', '--debug', action="store_true",
                    help="Enter debug mode to display frames per second")

args = parser.parse_args()
debug_mode = args.debug
show_preview = not args.no_preview
save_video = not args.no_video
use_camera = not args.file
if not use_camera:
    source = Path(args.file).resolve().absolute()
    if save_video:
        warnings.warn('Can not save video when not running camera.')
timeout = args.minutes


# Define monitoring class for treatment sessions
class Session:
    def __init__(self):
        print('Starting session...')

        # Init local storage
        self.label = f"session-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        self.store = Path('sessions', self.label).resolve().absolute()
        os.makedirs(str(self.store))

        # Prune oldest sessions to maximum number allowed
        sessions = sorted([Path(f) for f in glob.glob('sessions/session-*/')])
        max_sessions = 5
        while len(sessions) > max_sessions:
            oldest = sessions.pop(0)
            shutil.rmtree(oldest)

        # Init neural network models
        if use_camera:
            self.models = {'face_detect': 'face-detection-retail-0005-fp16-300x300-6shave.blob',
                           'head_pose': 'hopenet-faster-fp16-224x224-6shave.blob'}
        else:  # use all shaves if camera free
            self.models = {'face_detect': 'face-detection-retail-0005-fp16-300x300-8shave.blob',
                           'head_pose': 'hopenet-faster-fp16-224x224-8shave.blob'}
        self.model_sizes = {'face_detect': (300, 300),
                            'head_pose': (224, 224)}

        # Init camera
        self.preview_size = (1080, 1080)  # preferred square size for display
        self.pipeline = self.create_pipeline()
        print("Starting camera pipeline...")
        self.device = dai.Device()
        self.device.setLogLevel(dai.LogLevel.WARN)
        self.device.setLogOutputLevel(dai.LogLevel.WARN)
        self.running = self.device.startPipeline(self.pipeline)

        # Init streaming
        if use_camera:
            self.stream = self.device.getOutputQueue('camera_out', maxSize=1, blocking=False)
            if save_video:
                self.video = self.device.getOutputQueue('video_out', maxSize=30, blocking=True)
                self.v_file = open(self.store / 'video.h265', 'wb')
        else:
            self.source = cv2.VideoCapture(str(source))
            self.video_in = self.device.getInputQueue("video_in")

        # Init host processing queues
        self.frame_q = queue.Queue(maxsize=30)
        self.timestamp_q = queue.Queue(maxsize=30)
        self.face_q = self.device.getOutputQueue("face_detect", maxSize=4, blocking=False)
        self.pose_q = self.device.getOutputQueue("head_pose", maxSize=4, blocking=False)

        # Init tracking metrics
        self.found_faces = []
        self.start_time = datetime.datetime.now()
        self.threads = []
        self.data = {}
        self.meta = {'date_time': self.label,
                     'preview_size': self.preview_size,
                     'session_time': f"{timeout} min"}
        if debug_mode:
            self.fps = FPS()
            self.fps.start()
            print('Debug mode on.')

        time.sleep(1)  # wait for camera to boot
        if use_camera:
            print('Camera stream started.')
        else:
            print('Video stream started.')

    def create_pipeline(self):
        print('Creating camera pipeline...')
        pipeline = dai.Pipeline()

        # Setup RGB camera to capture 1080P
        if use_camera:
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam.setInterleaved(False)
            raw_fps = 30  # full 30 fps
            cam.setFps(raw_fps)

            # Setup preview stream for display and inference
            cam_xout = pipeline.createXLinkOut()
            cam_xout.setStreamName('camera_out')
            cam.setPreviewKeepAspectRatio(True)
            cam.setPreviewSize(*self.preview_size)
            cam.preview.link(cam_xout.input)

            # Setup video encoder stream for storage
            if save_video:
                raw_enc = pipeline.createVideoEncoder()
                cam.video.link(raw_enc.input)
                raw_enc.setDefaultProfilePreset(cam.getResolutionSize(), raw_fps,
                                                dai.VideoEncoderProperties.Profile.H265_MAIN)
                vid_out = pipeline.createXLinkOut()
                vid_out.setStreamName("video_out")
                raw_enc.bitstream.link(vid_out.input)

        print('Video streams created.')

        # Setup neural networks for inference with intermediate image nodes
        image_nodes = {}

        # Create face detection nnet
        print('Creating face detection neural network...')
        blob_path = Path('blobs', self.models['face_detect']).resolve().absolute()
        model = pipeline.create(dai.node.MobileNetDetectionNetwork)
        model.setConfidenceThreshold(0.7)
        model.setBlobPath(str(blob_path))

        # Create image node to resize frame for face detection
        node = pipeline.create(dai.node.ImageManip)
        node.initialConfig.setResize(*self.model_sizes['face_detect'])
        node.initialConfig.setKeepAspectRatio(True)
        node.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
        node.out.link(model.input)
        image_nodes['face_detect'] = node

        # Send face detection bounding boxes to host for display
        model_xout = pipeline.createXLinkOut()
        model_xout.setStreamName('face_detect')
        model.out.link(model_xout.input)

        # Create camera script to configure head pose crop based on face detection
        # Hopenet input size is hardcoded
        script = pipeline.create(dai.node.Script)
        script.inputs['bboxes'].setBlocking(False)
        script.inputs['bboxes'].setQueueSize(4)
        model.out.link(script.inputs['bboxes'])
        script.setScript("""
while True:
    bboxes = node.io['bboxes'].get().detections
    for bbox in bboxes:
        cfg = ImageManipConfig()
        cfg.setCropRect(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
        cfg.setResize(224, 224)
        node.io['transform'].send(cfg)
""")

        # Create hopenet head pose estimation nnet
        print('Creating head pose estimation neural network...')
        blob_path = Path('blobs', self.models['head_pose']).resolve().absolute()
        model = pipeline.create(dai.node.NeuralNetwork)
        model.setBlobPath(str(blob_path))

        # Create image node to crop face in frame for head pose estimation
        node = pipeline.create(dai.node.ImageManip)
        node.initialConfig.setResize(*self.model_sizes['head_pose'])
        node.initialConfig.setKeepAspectRatio(True)
        node.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
        node.setWaitForConfigInput(False)
        node.out.link(model.input)
        script.outputs['transform'].link(node.inputConfig)
        image_nodes['head_pose'] = node

        # Feed frame input from preview or host to image nodes as needed
        if use_camera:
            cam.preview.link(image_nodes['face_detect'].inputImage)
            cam.preview.link(image_nodes['head_pose'].inputImage)
        else:
            vid_in = pipeline.createXLinkIn()
            vid_in.setStreamName('video_in')
            vid_in.out.link(image_nodes['face_detect'].inputImage)
            vid_in.out.link(image_nodes['head_pose'].inputImage)

        # Send head pose model output vectors to host to store
        model_xout = pipeline.createXLinkOut()
        model_xout.setStreamName('head_pose')
        model.out.link(model_xout.input)

        print('Neural networks created.')
        return pipeline

    @staticmethod
    def timestamp(timedelta):
        # Converts timedelta to numerical timestamp in milliseconds
        ms = round((timedelta.days * 24 * 60 * 60 + timedelta.seconds) * 1000 +
                   (timedelta.microseconds / 1000))
        return ms

    def still_running(self):
        # Check if camera should timeout
        elapsed_min = (datetime.datetime.now() - self.start_time).seconds / 60
        if elapsed_min > timeout:
            print(f'Timeout of {timeout} minutes elapsed, session ending.')
            self.running = False

        # Check if camera is still running
        if not use_camera:
            return self.running and self.source.isOpened()
        return self.running

    def grab_frame(self):
        # Queue up a new frame for display or inference
        while self.running:
            if use_camera:
                # Read frame from camera stream
                msg = self.stream.tryGet()
                if not msg:
                    return
                frame = msg.getCvFrame().astype(np.uint8)  # BGR format, [H, W, C]
            else:
                # Read frame from video source
                valid, frame = self.source.read()
                if not valid or frame is None:
                    return
                frame = cv2.resize(frame, self.preview_size).astype(np.uint8)  # BGR format

                # Queue up frame as input for nnet flattened and channel reordered [C, H, W]
                tensor = dai.NNData()
                tensor.setLayer('input', frame.transpose(2, 0, 1).flatten())
                self.video_in.send(tensor)

            # Return frame for preview display
            if debug_mode:
                self.fps.update()
            return frame

    def run_file_save(self):
        # Save queued encoder frames to video files
        while self.running:
            msg = self.video.tryGet()  # raw video
            if msg:
                msg.getData().tofile(self.v_file)

    def run(self):
        # Start threads
        self.threads = []
        if save_video:
            self.threads.append(threading.Thread(target=self.run_file_save))
        for thread in self.threads:
            thread.start()

        # Run loop
        while self.still_running():
            # Grab frame if any
            frame = self.grab_frame()
            if frame is None:
                continue

            # Capture timestamp of frame
            elapsed = datetime.datetime.now() - self.start_time
            timestamp = self.timestamp(elapsed)

            # Get nnet output for frame
            msg = self.face_q.tryGet()

            if msg is not None:
                # Init data store
                data = {'fid': [],  # face id integer for frame
                        'bbx': [],  # face bounding box as ratio of frame
                        'ypr': []}  # yaw, pitch, roll angles in degrees

                # Loop over faces found
                for fid, det in enumerate(msg.detections):
                    data['fid'].append(fid)

                    # Convert bounding box to pixels
                    bbox = np.clip(np.array((det.xmin,
                                             det.ymin,
                                             det.xmax,
                                             det.ymax)), 0, 1)
                    data['bbx'].append(bbox.tolist())  # store as proportion of frame
                    bbox[0::2] *= frame.shape[1]  # x pixels
                    bbox[1::2] *= frame.shape[0]  # y pixels
                    bbox = bbox.astype(int)

                    # Get pose and read head pose angles, should always be output if face found
                    pose = self.pose_q.get()  # blocking
                    yaw_name, pitch_name, roll_name = pose.getAllLayerNames()
                    yaw = np.array(pose.getLayerFp16(yaw_name)).flatten()
                    pitch = np.array(pose.getLayerFp16(pitch_name)).flatten()
                    roll = np.array(pose.getLayerFp16(roll_name)).flatten()

                    # Calculate angle in degrees using softmax approach
                    # Ref: https://github.com/natanielruiz/deep-head-pose/blob/master/code/test_hopenet.py
                    # Softmax weights the angles by their probability bins
                    # Scale and center to -99 to +99 degree output range
                    yaw = np.sum(softmax(yaw) * np.arange(66)) * 3 - 99
                    pitch = np.sum(softmax(pitch) * np.arange(66)) * 3 - 99
                    roll = np.sum(softmax(roll) * np.arange(66)) * 3 - 99
                    data['ypr'].append([yaw, pitch, roll])

                    # Draw bounding box on frame
                    cv2.rectangle(frame,
                                  (bbox[0], bbox[1]),
                                  (bbox[2], bbox[3]),
                                  color=(0, 255, 0),
                                  thickness=1)

                    # Draw pose unit vectors on frame
                    origin = (int((bbox[0] + bbox[2]) / 2),
                              int((bbox[1] + bbox[3]) / 2))  # x, y
                    unit_len = int((bbox[2] - bbox[0]) / 2)  # half width of bbox
                    unit_pts = [None, None, None]  # init
                    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # RGB
                    line_widths = [3, 3, 2]  # front line thinner

                    # Convert angles to radians
                    yaw = -(yaw * np.pi / 180)
                    pitch = pitch * np.pi / 180
                    roll = roll * np.pi / 180

                    # X axis unit vector pointing to right in red
                    unit_pts[0] = [int(origin[0] + unit_len * (cos(yaw) * cos(roll))),
                                   int(origin[1] + unit_len * (cos(pitch) * sin(roll) +
                                                               cos(roll) * sin(pitch) * sin(yaw)))]

                    # Y axis unit vector pointing to up in green
                    unit_pts[1] = [int(origin[0] + unit_len * (-cos(yaw) * sin(roll))),
                                   int(origin[1] + unit_len * (cos(pitch) * cos(roll) -
                                                               sin(pitch) * sin(yaw) * sin(roll)))]

                    # Z axis unit vector pointing out of screen in blue
                    unit_pts[2] = [int(origin[0] + unit_len * (sin(yaw))),
                                   int(origin[1] + unit_len * (-cos(yaw) * sin(pitch)))]

                    # Draw vectors
                    for pt, c, w in zip(unit_pts, colors, line_widths):
                        cv2.line(frame, origin, tuple(pt), color=c, thickness=w)

                # Save all frame nnet output at timestamp if any
                if len(data['fid']):
                    self.data[timestamp] = data

            # Display preview if requested
            if show_preview:
                cv2.imshow("Preview", frame)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    self.running = False

        self.finish()
        self.write_json()
        print('Session ended, ready for upload.')

    def finish(self):
        # Display debug info
        if debug_mode:
            self.fps.stop()
            fps_str = f"{self.fps.fps():.2f}"
            self.meta['average_fps'] = fps_str
            print(f"Average FPS：{fps_str}")

        # Close all threads and stop inference
        for thread in self.threads:
            thread.join(2)
            if thread.is_alive():
                break
        print('Camera stream ended.')

        # Close open video files
        if use_camera and save_video:
            self.v_file.close()
            print('Video files written to disk.')

        # Close all opencv windows and files
        cv2.destroyAllWindows()
        if not use_camera:
            self.source.release()

    # TODO WIP
    def write_json(self):
        # Remove data points with no face
        valid_data = {}
        for ts, data in self.data.items():
            if data is not None:
                valid_data[ts] = data

        # Dump data and meta data to file
        with open(self.store / 'data.txt', 'w') as file:
            json.dump(valid_data, file)
        with open(self.store / 'meta.txt', 'w') as file:
            json.dump(self.meta, file)
        print('Data file written to disk.')

        # Copy to upload folder, will be deleted when uploaded
        if not debug_mode:
            dest = Path('upload', self.label).resolve().absolute()
            shutil.copytree(self.store, dest)


if __name__ == '__main__':
    sess = Session()

    # Register a graceful CTRL+C shutdown
    def shutdown(sig, frame):
        sess.running = False
    signal.signal(signal.SIGINT, shutdown)

    sess.run()
