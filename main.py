#!/usr/bin/env python3

"""
Submatter head movement tracker
Matt MacDonald, Subhash Talluri 2021

Main camera controller
1. Starts a video stream, finds and labels face, estimates head pose angles and positions, calculates velocities
2. Logs all data in json for upload
2. Shows annotated low res preview stream
3. Stores annotated high res video for subsequent analysis on the cloud

Pre-Trained Models
__________________
Face detection and identification:
OpenVINO model zoo - https://docs.openvinotoolkit.org/latest/omz_models_group_intel.html
face-detection-retail-0005
face-reidentification-retail-0095

Head pose estimation:
Hopenet-lite - https://github.com/OverEuro/deep-head-pose-lite

Future - Eye gaze estimation:
gaze-estimation-adas-0002
"""

# TODO use disparity depth calculation to add 3D space coordinate for head bbox
# TODO add eye gaze estimation
# TODO optional: add rectified disparity video file saving


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

from math import cos, sin, degrees
from scipy.spatial.distance import cosine
from scipy.special import softmax
from imutils.video import FPS

# Set arguments for running from CLI
parser = argparse.ArgumentParser()
parser.add_argument('-pre', '--preview', action="store_true",
                    help="Display annotated real time preview window")
parser.add_argument('-vid', '--video', type=str,
                    help="Path to video file to be used for inference")
parser.add_argument('-min', '--minutes', type=int, default=2,
                    help="Session time out in minutes (default 2 mins)")
parser.add_argument('-ano', '--annotate', action="store_true",
                    help="Save annotated preview video to h265 when using camera")
parser.add_argument('-sav', '--save', action="store_true",
                    help="Save raw source video to h265 in 1080P when using camera")
parser.add_argument('-deb', '--debug', action="store_true",
                    help="Enter debug mode to display frames per second")
parser.add_argument('-adv', '--advanced', action="store_true",
                    help="Run advanced face id neural net instead of naive method")

args = parser.parse_args()
debug_mode = args.debug
show_preview = args.preview
save_video = args.save
save_annotated = args.annotate
advanced_ffid = args.advanced
use_camera = not args.video
if save_video or save_annotated:
    assert use_camera, 'Saving video from file not supported, must use camera.'
if not use_camera:
    source = Path(args.video).resolve().absolute()
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
        self.models = {'face_detect': 'face-detection-retail-0005-fp16-300x300-6shave.blob',
                       'face_id': 'face-reidentification-retail-0095-fp16-128x128-6shave.blob',
                       'head_pose': 'hopenet-fp16-224x224-6shave-ipFP16.blob'}
        self.model_sizes = {'face_detect': (300, 300),
                            'face_id': (128, 128),
                            'head_pose': (224, 224)}

        # Init camera
        self.preview_size = (600, 600)  # must be square aspect ratio for nnets
        if save_annotated:
            assert (self.preview_size[0] % 8 == 0) and (self.preview_size[1] % 8 == 0), \
                "Preview size must be multiple of 8 for video encoder."
        self.pipeline = self.create_pipeline()
        self.device = dai.Device()
        print("Starting camera pipeline...")
        self.running = self.device.startPipeline(self.pipeline)

        # Init streaming
        if save_annotated:
            self.encoder = self.device.getInputQueue('annotated_in', maxSize=30, blocking=True)
            self.annotated = self.device.getOutputQueue('annotated_out', maxSize=30, blocking=True)
            self.a_file = open(self.store / 'annotated.h265', 'wb')
        if save_video:
            self.video = self.device.getOutputQueue('video_out', maxSize=30, blocking=True)
            self.v_file = open(self.store / 'video.h265', 'wb')
        if use_camera:
            self.stream = self.device.getOutputQueue('camera_out', maxSize=1, blocking=False)
        else:
            self.source = cv2.VideoCapture(str(source))

        # Init neural net inference queues
        self.face_q = self.device.getInputQueue("face_detect_in")
        if advanced_ffid:
            self.ffid_q = self.device.getInputQueue("face_id_in")
        self.pose_q = self.device.getInputQueue("head_pose_in")

        # Init host processing queues
        self.frame_q = queue.Queue(maxsize=30)
        self.timestamp_q = queue.Queue(maxsize=30)
        if save_annotated:
            self.message_q = queue.Queue(maxsize=30)

        # Init frame face counters for syncing
        self.face_done = False
        self.faces_to_ffid = 0
        self.faces_to_pose = 0

        # Init tracking metrics
        self.found_faces = []
        self.start_time = datetime.datetime.now()
        self.threads = []
        self.data = {}
        self.meta = {'date_time': self.label,
                     'preview_size': self.preview_size,
                     'session_time': f"{timeout} min",
                     'advanced_id': advanced_ffid}
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
            cam = pipeline.createColorCamera()
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam.setInterleaved(False)
            cam.setFp16(False)  # TODO test, should be uint8
            raw_fps = 30  # full 30 fps
            cam.setFps(raw_fps)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam_xout = pipeline.createXLinkOut()
            cam_xout.setStreamName('camera_out')

            # Setup preview stream for display and inference
            cam.setPreviewKeepAspectRatio(True)  # TODO test, should not squish
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

            # Setup annotated video encoder stream
            # TODO: test
            if save_annotated:
                ano_enc = pipeline.createVideoEncoder()
                ano_xin = pipeline.createXLinkIn()
                ano_xin.setStreamName('annotated_in')
                ano_xin.out.link(ano_enc.input)
                ano_fps = 3  # lower due to processing TODO: test
                ano_enc.setDefaultProfilePreset(*self.preview_size, ano_fps,
                                                dai.VideoEncoderProperties.Profile.H265_MAIN)
                ano_out = pipeline.createXLinkOut()
                ano_out.setStreamName("annotated_out")
                ano_enc.bitstream.link(ano_out.input)

        print('Video streams created.')

        # Setup neural networks for inference
        for name, blob in self.models.items():
            # Skip if not needed
            if not advanced_ffid and (name == 'face_id'):
                continue

            # Create nnet
            blob_path = Path('models', blob).resolve().absolute()
            model = pipeline.createNeuralNetwork()
            model.setBlobPath(str(blob_path))

            # Feed model input from host to nnet
            model_xin = pipeline.createXLinkIn()
            model_xin.setStreamName(f"{name}_in")
            model_xin.out.link(model.input)

            # Capture model output at host
            model_xout = pipeline.createXLinkOut()
            model_xout.setStreamName(name)
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
        if use_camera:
            # Read frame from camera stream
            msg = self.stream.tryGet()
            if not msg:
                return
            frame = msg.getCvFrame().astype(np.uint8)  # BGR format
        else:
            # Read frame from video source
            valid, frame = self.source.read()
            if not valid or frame is None:
                return
            frame = cv2.resize(frame, self.preview_size).astype(np.uint8)  # BGR format

        # Get elapsed time from host side
        elapsed = datetime.datetime.now() - self.start_time

        # Queue up frame for further processing
        self.frame_q.put(frame)
        if save_annotated:
            self.message_q.put(msg)  # needed for sending annotated frame back
        if debug_mode:
            self.fps.update()

        # Send to face detection nnet flattened and channel reordered [C, H, W]
        # Data type is uint8
        tensor = dai.NNData()
        frame = cv2.resize(frame, self.model_sizes['face_detect'])
        tensor.setLayer('data', frame.transpose(2, 0, 1).flatten().tolist())
        self.face_q.send(tensor)

        # Capture timestamp of frame and init data dict
        timestamp = self.timestamp(elapsed)
        self.timestamp_q.put(timestamp)
        self.data[timestamp] = {'face_boxes': [],
                                'face_ids': [],
                                # 'head_coords': [],
# TODO implement https://docs.luxonis.com/projects/api/en/latest/samples/spatial_location_calculator/
                                'head_angles': []}

    def run_face_detect(self):
        face_detect_q = self.device.getOutputQueue('face_detect', maxSize=30, blocking=True)

        while self.running:
            # Assure new frame in queue
            if self.face_done:
                continue

            # Get nnet output
            msg = face_detect_q.tryGet()
            if not msg:
                continue

            # Peek at timestamp and frame
            timestamp = self.timestamp_q.queue[0]
            frame = self.frame_q.queue[0]

            # Filter for bounding boxes with > 70 % confidence
            bboxes = np.array(msg.getFirstLayerFp16())
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]

            # Send cropped faces to nnets
            max_bbox = 3  # limit number of faces
            if not bboxes.size:
                self.data[timestamp] = None  # no valid data for frame
            for face_id, raw_bbox in enumerate(bboxes[:max_bbox]):
                # Convert bounding box coordinates to pixels
                y_size, x_size = frame.shape[:2]
                bounds = np.array([x_size, y_size, x_size, y_size])
                raw_bbox = np.clip(raw_bbox, 0, 1)
                bbox = (raw_bbox * bounds).astype(np.int)

                # Square box for cropping
                def scale_up(p_min, p_max, p_size, delta):
                    # Scales up min/max by delta equally on each side
                    p_min -= int(delta / 2)
                    p_max += int(delta / 2) + (delta % 2)  # if odd
                    # Offset if out of bounds
                    if p_min < 0:
                        p_max += -p_min
                        p_min = 0
                    elif p_max > p_size:
                        p_min -= p_max - p_size
                        p_max = p_size
                    return p_min, p_max

                x_len = bbox[2] - bbox[0]
                y_len = bbox[3] - bbox[1]
                if x_len < y_len:
                    bbox[0], bbox[2] = scale_up(bbox[0], bbox[2], x_size, y_len - x_len)
                elif y_len < x_len:
                    bbox[1], bbox[3] = scale_up(bbox[1], bbox[3], y_size, x_len - y_len)
                assert (bbox[2] - bbox[0]) == (bbox[3] - bbox[1]), 'Bounding box can not be made square.'

                # Crop frame for input to nnets
                bbox_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                # Send to face identification nnet
                # Data type is uint8
                if advanced_ffid:
                    tensor = dai.NNData()
                    bbox_frame = cv2.resize(bbox_frame, self.model_sizes['face_id'])
                    tensor.setLayer('data', bbox_frame.transpose(2, 0, 1).flatten().tolist())
                    self.ffid_q.send(tensor)

                # Resize, reorder [C x H x W] and normalize channels to ImageNet distribution per paper
                # Data type is fp16
                ideal_mean = [0.485, 0.456, 0.406]
                ideal_std = [0.229, 0.224, 0.225]
                bbox_frame = cv2.resize(bbox_frame, self.model_sizes['head_pose'])
                bbox_frame = bbox_frame.transpose(2, 0, 1)
                bbox_norm = bbox_frame.astype(np.float16)
                for c in range(bbox_frame.shape[0]):
                    bbox_norm[c] = ((bbox_norm[c] - bbox_frame[c].mean())
                                    / bbox_frame[c].std()) * ideal_std[c] + ideal_mean[c]

                # Send to head pose nnet
                tensor = dai.NNData()
                tensor.setLayer('data', bbox_norm.flatten().tolist())
                self.pose_q.send(tensor)

                # Store pixel based bounding box
                self.data[timestamp]['face_boxes'].append(bbox.tolist())

                # Store face ID number if naive method
                if not advanced_ffid:
                    self.data[timestamp]['face_ids'].append(face_id)

                # Increment sync counters
                if advanced_ffid:
                    self.faces_to_ffid += 1
                self.faces_to_pose += 1

            # Indicate frame complete
            self.face_done = True

    def run_face_id(self):
        face_id_q = self.device.getOutputQueue('face_id', maxSize=30, blocking=True)

        while self.running:
            # Assure new frame in queue
            if self.faces_to_ffid == 0:
                continue

            # Get nnet output
            msg = face_id_q.tryGet()
            if not msg:
                continue

            # Peek at timestamp
            timestamp = self.timestamp_q.queue[0]

            # Read face vector and assign ID number from found faces
            face_vec = np.array(msg.getFirstLayerFp16()).flatten()
            face_id = None
            threshold = 0.5  # TODO verify
            for i, ref_vec in enumerate(self.found_faces):
                if cosine(ref_vec, face_vec) < threshold:
                    face_id = i
                    break
            if face_id is None:
                # New face found
                face_id = len(self.found_faces)
                self.found_faces.append(face_vec)

            # Store face ID number
            self.data[timestamp]['face_ids'].append(face_id)

            # Decrement counter when frame complete
            self.faces_to_ffid -= 1

    def run_head_pose(self):
        head_pose_q = self.device.getOutputQueue('head_pose', maxSize=30, blocking=True)

        while self.running:
            # Assure new frame in queue
            if self.faces_to_pose == 0:
                continue

            # Get nnet output
            msg = head_pose_q.tryGet()
            if not msg:
                continue

            # Peek at timestamp
            timestamp = self.timestamp_q.queue[0]

            # Read head pose angles
            yaw_name, pitch_name, roll_name = msg.getAllLayerNames()
            yaw = np.array(msg.getLayerFp16(yaw_name)).flatten()
            pitch = np.array(msg.getLayerFp16(pitch_name)).flatten()
            roll = np.array(msg.getLayerFp16(roll_name)).flatten()

            # Calculate angle in degrees using softmax approach
            # Ref: https://github.com/natanielruiz/deep-head-pose/blob/master/code/test_hopenet.py
            # Softmax weights the angles by their probability bins
            # Scale and center to -99 to +99 degree output range
            yaw = np.sum(softmax(yaw) * np.arange(66)) * 3 - 99
            pitch = np.sum(softmax(pitch) * np.arange(66)) * 3 - 99
            roll = np.sum(softmax(roll) * np.arange(66)) * 3 - 99

            # Store head angles
            self.data[timestamp]['head_angles'].append([yaw, pitch, roll])

            # Decrement counter when frame complete
            self.faces_to_pose -= 1

    def run_file_save(self):
        # Save queued encoder frames to video files
        while self.running:
            if save_annotated:
                msg = self.annotated.tryGet()  # annotated
                if msg:
                    msg.getData().tofile(self.a_file)
            if save_video:
                msg = self.video.tryGet()  # raw video
                if msg:
                    msg.getData().tofile(self.v_file)

    def run(self):
        # Queue up an initial frame
        self.grab_frame()

        # Start neural network threads
        self.threads = [threading.Thread(target=self.run_face_detect),
                        threading.Thread(target=self.run_head_pose)]
        if advanced_ffid:
            self.threads.append(threading.Thread(target=self.run_face_id))
        if save_annotated or save_video:
            self.threads.append(threading.Thread(target=self.run_file_save))
        for thread in self.threads:
            thread.start()

        # Run loop
        while self.still_running():
            # Wait for nnets to sync
            if not self.face_done or (self.faces_to_ffid + self.faces_to_pose) > 0:
                continue

            # Queue up a new frame for inference
            self.grab_frame()

            # Consume frame and timestamp from queue
            frame = self.frame_q.get()
            timestamp = self.timestamp_q.get()

            # Indicate new frame ready for nnets
            self.face_done = False
            if advanced_ffid:
                self.faces_to_ffid = 0
            self.faces_to_pose = 0

            # Annotate video stream with nnet data results
            data = self.data[timestamp]  # face_boxes, face_ids, head_coords, head_angles

            # Draw bounding boxes and pose unit vectors
            if data is not None:
                for bbox, angles in zip(data['face_boxes'], data['head_angles']):
                    # Bounding box
                    cv2.rectangle(frame,
                                  (bbox[0], bbox[1]),
                                  (bbox[2], bbox[3]),
                                  color=(0, 255, 0),
                                  thickness=1)

                    # Unit vectors
                    origin = (int((bbox[0] + bbox[2]) / 2),
                              int((bbox[1] + bbox[3]) / 2))  # x, y
                    unit_len = int((bbox[2] - bbox[0]) / 2)  # half width of bbox
                    unit_pts = [None, None, None]  # init
                    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # RGB
                    line_widths = [3, 3, 2]  # front line thinner

                    # Convert angles to radians
                    yaw, pitch, roll = angles
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

            # Send annotated frame to be encoded to video
            if save_annotated:
                msg = self.message_q.get()
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # try grayscale
                msg.setType(dai.RawImgFrame.Type.YUV400p)
                msg.setData(grey.flatten().tolist())
                self.encoder.send(msg)

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
        if save_annotated:
            self.a_file.close()
        if save_video:
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
