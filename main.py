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

# TODO add hopelite head pose (openvino for blob)
# TODO add non debug mode start and stop control (command line over ssh)
# TODO sync up bbox and angle meta better to reduce amount of NaN
# TODO use disparity depth calculation to add 3D space coordinate for head bbox
# TODO filter to track one bbox only or track individuals
# TODO add a confidence filter for head pose and store in json meta
# TODO add eye gaze estimation
# TODO optional: add rectified disparity video file saving


import os
import datetime
from pathlib import Path

import argparse
import warnings

import queue
import threading
import json

import math
import cv2
import numpy as np
import depthai as dai

import ffmpeg
from imutils.video import FPS

# Set arguments for running from CLI
parser = argparse.ArgumentParser()
parser.add_argument('-pre', '--preview', action="store_true",
                    help="Display annotated real time preview window")
parser.add_argument('-vid', '--video', type=str,
                    help="Path to video file to be used for inference")
parser.add_argument('-min', '--minutes', type=int, default=2,
                    help="Session time out in minutes (default 2 mins)")
parser.add_argument('-sav', '--save', action="store_true",
                    help="Save raw source video in 1080P when using OAK-D")

args = parser.parse_args()
want_preview = args.preview
save_video = args.save
use_camera = not args.video
if not use_camera:
    source = Path(args.video).resolve().absolute()
timeout = args.minutes

# Define monitoring class for treatment sessions
class Session:
    def __init__(self, debug=False):
        print('Starting session...')
        self.debug = debug

        # Init local storage
        self.label = f"session-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        self.store = Path('sessions', self.label).resolve().absolute()
        os.makedirs(str(self.store))
        self.meta = {}
        # TODO upload folder and old file handling

        # Init neural network models
        self.models = {'face_detect': 'face-detection-retail-0005.blob',
                       'face_id': 'face-reidentification-retail-0095.blob',
                       'head_pose': 'hopenet-lite.blob'}
        self.model_sizes = {'face_detect': (300, 300),
                            'face_id': (128, 128),
                            'head_pose': (224, 224)}

        # Init camera
        self.preview_size = (600, 600)  # must be square aspect ratio for nnets
        self.pipeline = self.create_pipeline()
        self.device = dai.Device(self.pipeline)
        print("Starting camera pipeline...")
        self.running = self.device.startPipeline()

        # Init streaming
        self.annotated = self.device.getOutputQueue('annotated_out', maxSize=30, blocking=True)
        self.a_file = open(self.store / 'annotated.h265', 'wb')
        if save_video:
            self.video = self.device.getOutputQueue('cam_enc', maxSize=30, blocking=True)
            self.v_file = open(self.store / 'video.h265', 'wb')
        if use_camera:
            self.stream = self.device.getOutputQueue('cam_out', maxSize=1, blocking=False)
        else:
            self.source = cv2.VideoCapture(str(source))

        # Init neural net inference queues
        self.face_q = self.device.getInputQueue("face_detect_in")
        self.reid_q = self.device.getInputQueue("face_id_in")
        self.pose_q = self.device.getInputQueue("head_pose_in")

        # Init host processing queues
        self.frme_q = queue.Queue(maxsize=30)
        self.time_q = queue.Queue(maxsize=30)
        self.bbox_q = queue.Queue(maxsize=30)
        self.face_q = queue.Queue(maxsize=30)
        self.angl_q = queue.Queue(maxsize=30)

        # Init tracking metrics
        time.sleep(1)  # wait for camera to boot
        self.start_time = datetime.datetime.now()
        self.threads = []
        if self.debug:
            self.fps = FPS()
            self.fps.start()
            print('Debug mode on.')
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
            cam.setInterleaved(False)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam_xout = pipeline.createXLinkOut()
            cam_xout.setStreamName('cam_out')

            # Setup preview stream for display and inference
            cam.setPreviewSize(*self.preview_size)
            cam.preview.link(cam_xout.input)

            # Setup video encoder stream for storage
            if save_video:
                raw_enc = pipeline.createVideoEncoder()
                cam.video.link(raw_enc.input)
                raw_fps = 30  # full 30 fps
                raw_enc.setDefaultProfilePreset(cam.getResolutionSize(), raw_fps,
                                                dai.VideoEncoderProperties.Profile.H265_MAIN)
                cam_enc = pipeline.createXLinkOut()
                cam_enc.setStreamName("cam_enc")
                raw_enc.bitstream.link(cam_enc.input)

        # Setup annotated video encoder stream
        # TODO: test
        anno_enc = pipeline.createVideoEncoder()
        anno_xin = pipeline.createXLinkIn()
        anno_xin.setStreamName('annotated_in')
        anno_xin.link(anno_enc.input)
        anno_fps = 25  # lower due to processing
        anno_enc.setDefaultProfilePreset(self.preview_size, anno_fps,
                                         dai.VideoEncoderProperties.Profile.H265_MAIN)
        anno_xout = pipeline.createXLinkOut()
        anno_xout.setStreamName("annotated_out")
        anno_enc.bitstream.link(anno_xout.input)
        print('Video streams created.')

        # Setup neural networks for inference
        for name, blob in self.models.items():
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

    @staticmethod
    # TODO needed???
    def to_tensor_result(packet):
        return {
            tensor.name: np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
            for tensor in packet.getRaw().tensors
        }

    def still_running(self):
        # Check if camera should timeout
        elapsed_min = (datetime.datetime.now() - self.start_time).seconds / 60
        if elapsed_min > timeout:
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
                break
            frame = msg.getCvFrame().astype(np.uint8)  # BGR format
            elapsed = msg.getTimestamp()
            self.frme_q.put(frame)
            print(frame.shape)  # TODO delete

        else:
            # Read frame from video source
            valid, frame = self.source.read()
            if not valid or frame is None:
                break
            frame = cv2.resize(frame, self.preview_size).astype(np.uint8)  # BGR format
            elapsed = datetime.datetime.now() - self.start_time

        # Queue up frame for further processing
        self.frme_q.put(frame)
        if self.debug:
            self.fps.update()

        # Send frame data to nnet flattened and channel reordered
        frame_data = depthai.NNData()
        frame = cv2.resize(frame, self.model_sizes['face_detect'])
        frame_data.setLayer('data', list(frame.transpose(2, 0, 1).ravel()))
        self.face_q.send(frame_data)

        # Capture timestamp of frame and init data dict
        timestamp = self.timestamp(elapsed)
        self.time_q.put(timestamp)
        self.meta[timestamp] = {}

    # TODO WIP
    def run_face_detect(self):
        face_detect_q = self.device.getOutputQueue('face_detect')

        while self.running:
            # Get nnet output
            msg = face_detect_q.tryGet()
            if not msg:
                continue

            # Peek at frame and timestamp
            frame = self.frme_q.queue[0].copy()
            timestamp = self.time_q.queue[0]

            # Filter for bounding boxes with > 70 % confidence
            bboxes = np.array(msg.getFirstLayerFp16())
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]

            for raw_bbox in bboxes:
                # Convert bounding box coordinates to pixels
                y_size, x_size = frame.shape[:2]
                bounds = np.array([x_size, y_size, x_size, y_size])
                raw_bbox = np.clip(raw_bbox, 0, 1)
                bbox = (raw_bbox * bounds).astype(np.int)

                # Store bounding box position as proportion of frame
                self.meta[self.timestamp]['bbox_coords'] = raw_bbox.tolist()

                # Queue up cropped frame for input to landmark NN
                bbox_frame = self.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                bbox_data = cv2.resize(bbox_frame, (160, 160)).transpose(2, 0, 1)
                bbox_data = bbox_data.flatten().tolist()
                tensor = dai.NNData()
                tensor.setLayer('data', bbox_data)
                landmark_qin.send(tensor)
                # TODO get face id and store conf
                # Queue up bounding box for landmarking reference
                self.bbox_q.put(bbox)

    # TODO WIP
    def run_face_id(self):
        landmark_q = self.device.getOutputQueue('landmark', maxSize=4, blocking=False)

        while self.running:
            while self.bbox_q.qsize():
                # Retrieve landmark points
                msg = landmark_q.tryGet()
                if msg is None:
                    continue
                raw_pts = None
                for tensor in msg.getRaw().tensors:
                    if tensor.name == 'StatefulPartitionedCall/strided_slice_2/Split.0':
                        raw_pts = np.array(msg.getLayerFp16(tensor.name))

                # Convert coordinate points to full frame pixels
                face_bbox = self.bbox_q.get()
                y_size = face_bbox[3] - face_bbox[1]
                x_size = face_bbox[2] - face_bbox[0]
                bounds = np.array([x_size, y_size] * (raw_pts.size // 2))
                raw_pts = np.clip(raw_pts, 0, 1)
                pts = (raw_pts * bounds).astype(int)
                origin = np.array([face_bbox[0], face_bbox[1]] * (pts.size // 2))
                pts = pts + origin

                # Filter points to list of 13 key facial landmarks
                key_pts = self.key_landmarks(pts)

                # Calculate head pose vector from landmark points
                unit_pts, pitch, yaw, roll = self.head_pose(key_pts)
                angles = [pitch, yaw, roll]

                # Store pose vector and angles for current timestamp
                self.meta[self.timestamp]['pitch'] = pitch
                self.meta[self.timestamp]['yaw'] = yaw
                self.meta[self.timestamp]['roll'] = roll

                # Queue up for preview window display
                if self.debug:
                    self.face_q.put(face_bbox)
                    self.mark_q.put(key_pts)
                    self.unit_q.put(unit_pts)
                    self.pose_q.put(angles)

    # TODO WIP
    def run_head_pose(self, key_pts):
        # Fill in the 2D reference point, follow https://ibug.doc.ic.ac.uk/resources/300-W/
        # reprojectdst, _, pitch, yaw, roll = get_head_pose(np.array(self.hand_points))

        # World Coordinate System (UVW) 3D reference points
        # Face model reference http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
        ref_pts = np.float32([[6.825897, 6.760612, 4.402142],  # Upper left corner of left eyebrow
                              [1.330353, 7.122144, 6.903745],  # Left eyebrow right corner
                              [-1.330353, 7.122144, 6.903745],  # Right eyebrow left corner
                              [-6.825897, 6.760612, 4.402142],  # Upper right corner of right eyebrow
                              [5.311432, 5.485328, 3.987654],  # Upper left corner of left eye
                              [1.789930, 5.393625, 4.413414],  # Upper right corner of left eye
                              [-1.789930, 5.393625, 4.413414],  # Upper left corner of right eye
                              [-5.311432, 5.485328, 3.987654],  # Upper right corner of right eye
                              [2.005628, 1.409845, 6.165652],  # Upper left corner of nose
                              [-2.005628, 1.409845, 6.165652],  # Upper right corner of nose
                              [2.774015, -2.080775, 5.048531],  # Upper left corner of mouth
                              [-2.774015, -2.080775, 5.048531],  # Upper right corner of mouth
                              [0.000000, -3.116408, 6.097667],  # Lower corner of mouth
                              [0.000000, -7.415691, 4.070434]])  # Chin angle

        # Calculate the rotation and translation vectors
        img_pts = np.float32(key_pts)
        _, rotation, translation = cv2.solvePnP(ref_pts, img_pts, self.K, self.D)

        # Project unit vectors to visually display head pose
        length = 10  # non dynamic
        unit_pts = length * np.float32([(0, 0, 0), (1, 0, 0), (0, -1, 0), (0, 0, 1)])
        unit_pts, _ = cv2.projectPoints(unit_pts, rotation, translation, self.K, self.D)
        unit_pts = list(map(tuple, unit_pts.reshape(4, 2)))

        # Calculate Euler angle
        rotation_v = cv2.Rodrigues(rotation)[0]  # convert rotation matrix to a rotation vector
        pose_matrix = cv2.hconcat((rotation_v, translation))
        euler_angle = cv2.decomposeProjectionMatrix(pose_matrix)[-1]

        # Convert Euler angle into pitch, yaw and roll
        pitch, yaw, roll = [math.radians(angle) for angle in euler_angle]
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        return unit_pts, pitch, yaw, roll

    def run_file_saving(self):
        # Save queued encoder frames to video files
        msg = self.annotated.tryGet()  # annotated
        if msg:
            msg.getData().tofile(self.a_file)
        if save_video:
            msg = self.annotated.tryGet()  # raw video
            if msg:
                msg.getData().tofile(self.v_file)

    def run(self):
        # Start neural network threads
        self.threads = [
            threading.Thread(target=self.run_face_detect),
            threading.Thread(target=self.run_face_id),
            threading.Thread(target=self.run_head_pose)
            threading.Thread(target=self.run_file_saving)]
        for thread in self.threads:
            thread.start()

        # TODO thread annotated video and meta data
        # Run loop
        while self.still_running():
            # Queue up a frame for inference
            self.grab_frame()

            # Update preview window for display and annotation
            if self.debug:
                frame = self.frme_q.get()
                if self.face_q.qsize() and self.mark_q.qsize() and self.unit_q.qsize():
                    for i in range(self.face_q.qsize()):
                        face_bbox = self.face_q.get()
                        key_pts = self.mark_q.get()
                        unit_pts = self.unit_q.get()
                        angles = self.pose_q.get()

                        # Draw bounding box
                        cv2.rectangle(debug_frame,
                                      (face_bbox[0], face_bbox[1]),
                                      (face_bbox[2], face_bbox[3]),
                                      color=(0, 255, 0), thickness=1)

                        # Draw landmark points
                        for pt in key_pts:
                            cv2.circle(debug_frame, pt,
                                       radius=2, color=(255, 0, 0), thickness=1)

                        # Draw unit vectors
                        origin = unit_pts[0]
                        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
                        for pt, color in zip(unit_pts[1:], colors):
                            cv2.line(debug_frame, origin, pt, color=color, thickness=1)

                        # Add text to display angles
                        cv2.putText(debug_frame,
                                    "pitch:{:.2f}, yaw:{:.2f}, roll:{:.2f}".format(*angles),
                                    (face_bbox[0] - 30, face_bbox[1] - 30),
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                    fontScale=0.45, color=(255, 0, 0))
                # TODO Save annotated
                cv2.imshow("Preview", debug_frame)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    self.running = False

        self.finish()
        self.write_json()
        print('Session ended, ready for upload.')

    def finish(self):
        # Display debug info
        if self.debug:
            self.fps.stop()
            print(f"Average FPS：{self.fps.fps():.2f}")

        # Close all threads and stop inference
        for thread in self.threads:
            thread.join(2)
            if thread.is_alive():
                break
        print('Camera stream ended.')

        # Close open video files
        self.a_file.close()
        if save_video:
            self.v_file.close()
        print('Video files written to disk.')

        # Close all opencv windows and files
        cv2.destroyAllWindows()
        if not use_camera:
            self.source.release()

    def write_json(self):
        # Add UTC style timestamps
        # TODO

        # Remove empty data points
        meta = {}
        for key, val in self.meta.items():
            if val:
                meta[key] = val

        # Dump to file
        with open(self.store / 'meta.txt', 'w') as file:
            json.dump(meta, file)
        print('Data file written to disk.')


if __name__ == '__main__':
    sess = Session(debug=True, convert=True)
    sess.run()
