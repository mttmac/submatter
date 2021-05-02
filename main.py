#!/usr/bin/env python3

"""
Submatter head movement tracker
Matt MacDonald, Subhash Talluri 2021

Main camera controller
Starts a video stream, detects head pose, extracts vectors and logs in json for upload
Stores raw video and disparity for subsequent analysis on the cloud

Based on https://github.com/luxonis/depthai-experiments gen2-head-posture-detection experiment

Trained neural networks sourced from OpenVino toolkit:
face-detection-retail-0004 (SSD face detection)
Unknown blob!!! facial-landmarks-35-adas-0002 (CNN facial landmarks estimation)
Future:
head-pose-estimation-adas-0001 (CNN head pose estimation)
gaze-estimation-adas-0002 (VGG eye gaze estimation)
"""
# TODO use disparity depth calcualtion to improve accuracy of landmarking
# TODO add rectified disparity video file saving
# TODO Upgrade face detection to 0005
# TODO implement a known facial landmarks estimator
# TODO add tracking to faces for consistency


import datetime
import os
import json
from pathlib import Path

import math
import cv2
import numpy as np
import depthai as dai

from queue import Queue
import threading

from imutils.video import FPS

import time

def timer(function):
    """
    Decorator function timer
    :param function:The function you want to time
    :return:
    """

    def wrapper(*args, **kwargs):
        time_start = time.time()
        res = function(*args, **kwargs)
        cost_time = time.time() - time_start
        print("【 %s 】operation hours：【 %s 】second" % (function.__name__, cost_time))
        return res

    return wrapper


class Session:
    def __init__(self, debug=True):
        print('Starting session...')
        self.debug = debug

        # Init local storage
        self.label = f"session-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.store = Path('sessions', self.label).resolve().absolute()
        os.makedirs(str(self.store))
        self.meta = {}

        # Init camera
        self.device = dai.Device(self.create_pipeline())
        print("Starting camera pipeline...")
        self.running = self.device.startPipeline()
        self.video = self.device.getOutputQueue('cam_out', maxSize=30, blocking=True)

        # Init streaming
        self.stream = self.device.getOutputQueue('cam_preview', maxSize=1, blocking=False)
        self.stream_start = None
        self.threads = []
        if self.debug:
            self.fps = FPS()
            self.fps.start()

        # Init neural net inference queues
        self.frame = None
        self.timestamp = None
        self.bboxes = []
        self.bbox_q = Queue()
        self.face_q = Queue()
        self.mark_q = Queue()
        self.unit_q = Queue()

        # Init camera pixel coordinate system (xy): camera eigen and distortion coefficients
        # Camera coordinate system (XYZ): camera internal matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        self.K = np.float32([6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
                             0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
                             0.0, 0.0, 1.0]).reshape(3, 3)
        # Image center coordinate system (uv): camera distortion coefficients [k1, k2, p1, p2, k3]
        self.D = np.float32([7.0834633684407095e-002, 6.9140193737175351e-002,
                             0.0, 0.0, -1.3073460323689292e+000]).reshape(5, 1)

    @staticmethod
    def create_pipeline():
        print('Creating camera pipeline...')
        pipeline = dai.Pipeline()

        # Setup RGB camera to capture 1080P
        cam = pipeline.createColorCamera()
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)

        # Setup video encoder stream for storage
        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName('cam_out')
        enc = pipeline.createVideoEncoder()
        cam.video.link(enc.input)
        enc_fps = 25
        enc.setDefaultProfilePreset(cam.getResolutionSize(), enc_fps,
                                    dai.VideoEncoderProperties.Profile.H265_MAIN)
        enc.bitstream.link(cam_xout.input)

        # Setup preview stream for display and inference
        cam_xpre = pipeline.createXLinkOut()
        cam_xpre.setStreamName("cam_preview")
        cam.setPreviewSize(300, 300)
        cam.preview.link(cam_xpre.input)
        print('Video streams created.')

        # Setup neural networks for inference
        models = {'detect': 'face-detection-retail-0004_openvino_2020_1_4shave.blob',
                  'landmark': 'face_landmark_160x160_openvino_2020_1_4shave.blob'}
        first_model = 'detect'
        for name, blob in models.items():
            blob_path = Path('models', blob).resolve().absolute()
            model = pipeline.createNeuralNetwork()
            model.setBlobPath(str(blob_path))

            if name == first_model:
                # Feed video stream to nnet
                cam.preview.link(model.input)
            else:
                # Feed model output from host to nnet
                model_xin = pipeline.createXLinkIn()
                model_xin.setStreamName(f"{name}_in")
                model_xin.out.link(model.input)
            model_xout = pipeline.createXLinkOut()
            model_xout.setStreamName(name)
            model.out.link(model_xout.input)
        print('Neural networks created.')

        return pipeline

    def elapsed_ms(self, timedelta):
        ms = ((timedelta.days * 24 * 60 * 60 + timedelta.seconds) * 1000 +
              (timedelta.microseconds / 1000))
        if self.stream_start is None:
            self.stream_start = ms
        return ms - self.stream_start

    def grab_frame(self, retries=0):
        msg = self.stream.get()
        frame = msg.getFrame()
        self.frame = frame.transpose(1, 2, 0).astype(np.uint8)  # correct order and bitsize
        self.timestamp = self.elapsed_ms(msg.getTimestamp())
        self.meta[self.timestamp] = {}  # create empty dict for timestamp
        if self.debug:
            self.fps.update()

    def run_face_detection(self):
        detect_q = self.device.getOutputQueue('detect')
        landmark_qin = self.device.getInputQueue('landmark_in', maxSize=4, blocking=False)

        while self.running:
            if self.frame is None:
                continue
            # Get NN output and filter for bounding boxes with >70% confidence
            bboxes = np.array(detect_q.get().getFirstLayerFp16())
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            self.bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]

            # TODO filter to track one bbox only or track individuals
            # TODO add a confidence filter for landmarking as well
            for raw_bbox in self.bboxes:
                # Convert bounding box coordinates to pixels
                y_size, x_size = self.frame.shape[:2]
                bounds = np.array([x_size, y_size] * (raw_bbox.size // 2))
                raw_bbox = np.clip(raw_bbox, 0, 1)
                bbox = (raw_bbox * bounds).astype(int)

                # Queue up bounding box for landmarking reference
                self.bbox_q.put(bbox)

                # Store bounding box position as proportion of frame
                self.meta[self.timestamp]['bbox_coords'] = raw_bbox.tolist()

                # Queue up cropped frame for input to landmark NN
                bbox_frame = self.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                bbox_data = cv2.resize(bbox_frame, (160, 160)).transpose(2, 0, 1).flatten()
                tensor = dai.NNData().setLayer('data', bbox_data.tolist())
                landmark_qin.send(tensor)

    def run_face_landmarking(self):
        landmark_q = self.device.getOutputQueue('landmark', maxSize=4, blocking=False)

        while self.running:
            while self.bboxes:
                # Retrieve landmark points
                msg = landmark_q.get()
                raw_pts = None
                for tensor in msg.getRaw().tensors:
                    if tensor.name == 'StatefulPartitionedCall/strided_slice_2/Split.0':
                        raw_pts = np.array(msg.getLayerFp16(tensor.name))
                if raw_pts is None:
                    continue

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
                unit_pts, euler_angle, pitch, yaw, roll = self.head_pose(key_pts)

                # Store pose vector and angles for current timestamp
                self.meta[self.timestamp]['euler_angle'] = euler_angle
                self.meta[self.timestamp]['pitch'] = pitch
                self.meta[self.timestamp]['yaw'] = yaw
                self.meta[self.timestamp]['roll'] = roll

                # Queue up for preview window display
                if self.debug:
                    self.face_q.put(face_bbox)
                    self.mark_q.put(key_pts)
                    self.unit_q.put(unit_pts)

    @staticmethod
    def key_landmarks(pts):
        # Return key landmark points needed for pose estimation
        key_pts = [(pts[34], pts[35]),  # 17 Left eyebrow upper left corner
                   (pts[42], pts[43]),  # 21 Left eyebrow right corner
                   (pts[44], pts[45]),  # 22 Right eyebrow upper left corner
                   (pts[52], pts[53]),  # 26 Right eyebrow upper right corner
                   (pts[72], pts[73]),  # 36 Left eye upper left corner
                   (pts[78], pts[79]),  # 39 Left eye upper right corner
                   (pts[84], pts[85]),  # 42 Right eye upper left corner
                   (pts[90], pts[91]),  # 45 Upper right corner of the right eye
                   (pts[62], pts[63]),  # 31 Upper left corner of the nose
                   (pts[70], pts[71]),  # 35 Upper right corner of the nose
                   (pts[96], pts[97]),  # 48 Upper left corner
                   (pts[108], pts[109]),  # 54 Upper right corner of the mouth
                   (pts[114], pts[115]),  # 57 Lower central corner of the mouth
                   (pts[16], pts[17])]  # 8 Chin corner
        return key_pts

    def head_pose(self, key_pts):
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
        length = 10  # TODO make vector length sizing dynamic
        unit_pts = length * np.float32([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
        unit_pts, _ = cv2.projectPoints(unit_pts, rotation, translation, self.K, self.D)
        unit_pts = list(map(tuple, unit_pts.reshape(8, 2)))

        # Calculate Euler angle
        rotation_v = cv2.Rodrigues(rotation)[0]  # convert rotation matrix to a rotation vector
        pose_matrix = cv2.hconcat((rotation_v, translation))
        euler_angle = cv2.decomposeProjectionMatrix(pose_matrix)[-1]

        # Convert Euler angle into pitch, yaw and roll
        pitch, yaw, roll = [math.radians(angle) for angle in euler_angle]
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        return unit_pts, euler_angle, pitch, yaw, roll

    def run(self):
        # Start neural network threads
        self.threads = [threading.Thread(target=self.run_face_detection, daemon=True),
                        threading.Thread(target=self.run_face_landmarking, daemon=True)]
        for thread in self.threads:
            thread.start()

        # Run loop
        with open(self.store / 'session.h265', 'wb') as video:
            while self.running:
                # Grab frame for preview and inference
                self.grab_frame()

                # Save queued raw frames to video file
                try:
                    self.video.get().getData().tofile(video)
                except RuntimeError:
                    pass

                # Update preview window
                if self.debug:
                    debug_frame = self.frame.copy()
                    if self.face_q.qsize() and self.mark_q.qsize() and self.unit_q.qsize():
                        for i in range(self.face_q.qsize()):
                            face_bbox = self.face_q.get()
                            key_pts = self.mark_q.get()
                            unit_pts = self.unit_q.get()

                            # Draw bounding box
                            cv2.rectangle(debug_frame,
                                          (face_bbox[0], face_bbox[1]),
                                          (face_bbox[2], face_bbox[3]),
                                          color=(0, 255, 0), thickness=2)

                            # Draw landmark points
                            for pt in key_pts:
                                cv2.circle(debug_frame, pt,
                                           radius=2, color=(255, 0, 0), thickness=1)

                            # Draw unit vectors
                            origin = unit_pts[0]
                            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
                            for pt, color in zip(unit_pts[1:], colors):
                                cv2.line(debug_frame, origin, pt, color=color, thickness=2)

                            # Add text to display angles
                            meta = self.meta[self.timestamp]
                            angles = [meta['pitch'], meta['yaw'], meta['roll']]
                            cv2.putText(debug_frame,
                                        "pitch:{:.2f}, yaw:{:.2f}, roll:{:.2f}".format(*angles),
                                        (face_bbox[0] - 30, face_bbox[1] - 30),
                                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                        fontScale=0.45, color=(255, 0, 0))

                    cv2.imshow("Preview", debug_frame)
                    if cv2.waitKey(1) == ord('q'):
                        cv2.destroyAllWindows()
                        self.running = False
                # TODO add non debug break option

        self.finish()
        self.write_json()
        self.upload_data()
        print('Session ended.')

    def finish(self):
        if self.debug:
            self.fps.stop()
            print(f"Average FPS：{self.fps.fps():.2f}")

        # Close all streams and stop inference
        print('Closing camera pipeline...')
        cv2.destroyAllWindows()
        self.running = False
        for thread in self.threads:
            thread.join(2)
            if thread.is_alive():
                break

    def write_json(self):
        with open(self.store / 'meta.txt', 'w') as file:
            json.dump(self.meta, file)

    def upload_data(self):
        # TODO implement cloud upload
        pass


if __name__ == '__main__':
    sess = Session()
    # sess.run()
