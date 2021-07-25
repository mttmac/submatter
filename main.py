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
import time
import datetime
from pathlib import Path

import argparse
import signal
import warnings

import queue
import threading
import json

from math import cos, sin, degrees
import cv2
import numpy as np
import depthai as dai

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
parser.add_argument('-deb', '--debug', action="store_true",
                    help="Enter debug mode to display frames per second")

args = parser.parse_args()
debug_mode = args.debug
show_preview = args.preview
save_video = args.save
use_camera = not args.video
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
        self.data = {}
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
        self.encoder = self.device.getOutputQueue('annotated_in', maxSize=30, blocking=True)
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
        self.reid_q = self.device.getInputQueue("face_id_in")
        self.pose_q = self.device.getInputQueue("head_pose_in")

        # Init host processing queues and sync counters
        self.frame_q = queue.Queue(maxsize=30)
        self.timestamp_q = queue.Queue(maxsize=30)
        self.new = True  # new frame at front of queue
        self.face_n = 0
        self.reid_n = 0
        self.pose_n = 0

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
                cam_enc = pipeline.createXLinkOut()
                cam_enc.setStreamName("video_out")
                raw_enc.bitstream.link(cam_enc.input)

        # Setup annotated video encoder stream
        # TODO: test
        anno_enc = pipeline.createVideoEncoder()
        anno_xin = pipeline.createXLinkIn()
        anno_xin.setStreamName('annotated_in')
        anno_xin.link(anno_enc.input)
        anno_fps = 25  # lower due to processing TODO: test
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
                return
            frame = msg.getCvFrame().astype(np.uint8)  # BGR format
            elapsed = msg.getTimestamp()
            print(frame.shape)  # TODO delete

        else:
            # Read frame from video source
            valid, frame = self.source.read()
            if not valid or frame is None:
                return
            frame = cv2.resize(frame, self.preview_size).astype(np.uint8)  # BGR format
            elapsed = datetime.datetime.now() - self.start_time

        # Queue up frame for further processing
        self.frame_q.put(frame)
        if debug_mode:
            self.fps.update()

        # Send to face detection nnet flattened and channel reordered [C, H, W]
        tensor = dai.NNData()
        frame = cv2.resize(frame, self.model_sizes['face_detect'])
        tensor.setLayer('data', frame.transpose(2, 0, 1).flatten().tolist())
        self.face_q.send(tensor)

        # Capture timestamp of frame and init data dict
        timestamp = self.timestamp(elapsed)
        self.timestamp_q.put(timestamp)
        self.data[timestamp] = {'face_boxes': [],
                                'face_ids': [],
                                'head_coords': [],
                                'head_angles': []}

    def run_face_detect(self):
        face_detect_q = self.device.getOutputQueue('face_detect')

        while self.running:
            # Wait for new frame
            if not self.new:
                continue
            self.new = False

            # Get nnet output
            msg = face_detect_q.tryGet()
            if not msg:
                continue

            # Peek at frame and timestamp
            frame = self.frame_q.queue[0]  # TODO: need .copy()?
            timestamp = self.timestamp_q.queue[0]

            # Filter for bounding boxes with > 70 % confidence
            bboxes = np.array(msg.getFirstLayerFp16())
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]

            # Store bounding box position as proportion of frame
            max_bbox = 3  # limit number of faces
            self.data[timestamp]['face_boxes'] = [bb.tolist() for bb in bboxes[:max_bbox]]

            # Send cropped faces to nnets
            for raw_bbox in bboxes[:max_bbox]:
                # Convert bounding box coordinates to pixels
                y_size, x_size = frame.shape[:2]
                bounds = np.array([x_size, y_size, x_size, y_size])
                raw_bbox = np.clip(raw_bbox, 0, 1)
                bbox = (raw_bbox * bounds).astype(np.int)

                # Store head coordinate center as portion of frame, z unknown currently
                coord = [(raw_bbox[0] + raw_bbox[2]) / 2, (raw_bbox[1] + raw_bbox[3]) / 2, None]
                self.data[timestamp]['head_coords'].append(coord)

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
                tensor = dai.NNData()
                bbox_frame = cv2.resize(bbox_frame, self.model_sizes['face_id'])
                tensor.setLayer('data', bbox_frame.transpose(2, 0, 1).flatten().tolist())
                self.reid_q.send(tensor)

                # Send to head pose nnet
                tensor = dai.NNData()
                bbox_frame = cv2.resize(bbox_frame, self.model_sizes['head_pose'])
                tensor.setLayer('data', bbox_frame.transpose(2, 0, 1).flatten().tolist())
                self.pose_q.send(tensor)

                # Increment sync counter
                self.face_n += 1

    def run_face_id(self):
        face_id_q = self.device.getOutputQueue('face_id')

        while self.running:
            # Get nnet output
            msg = face_id_q.tryGet()
            if not msg:
                continue

            # Peek at frame and timestamp
            frame = self.frame_q.queue[0].copy()
            timestamp = self.timestamp_q.queue[0]

            # TODO WIP


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

    def run_head_locate(self):
        # Triangulate to estimate head position from camera in meters
        # TODO WIP

    def run_file_save(self):
        # Save queued encoder frames to video files
        msg = self.annotated.tryGet()  # annotated
        if msg:
            msg.getData().tofile(self.a_file)
        if save_video:
            msg = self.video.tryGet()  # raw video
            if msg:
                msg.getData().tofile(self.v_file)

    def run(self):
        # Start neural network threads
        self.threads = [
            threading.Thread(target=self.run_face_detect),
            threading.Thread(target=self.run_face_id),
            threading.Thread(target=self.run_head_pose),
            threading.Thread(target=self.run_head_locate),
            threading.Thread(target=self.run_file_save)]
        for thread in self.threads:
            thread.start()

        # Run loop
        while self.still_running():
            # Queue up a frame for inference
            self.grab_frame()

            # Assure frame in queue
            if self.frame_q.qsize() == 0:
                continue

            # Wait for nnets to sync
            if (self.reid_n < self.face_n) or (self.pose_n < self.face_n):
                continue

            # Consume frame and timestamp from queue
            frame = self.frame_q.get()
            timestamp = self.timestamp_q.get()
            self.new = True  # new frame ready for face detection

            # Annotate video stream with nnet data results
            data = self.data[timestamp]  # face_boxes, face_ids, head_coords, head_angles

            # Draw bounding boxes and pose unit vectors
            for bbox, angles in zip(data['face_boxes'], data['head_angles']):
                # Bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              color=(0, 255, 0), thickness=1)

                # Unit vectors
                origin = [int((bbox[0] + bbox[2]) / 2),
                          int((bbox[1] + bbox[3]) / 2)]  # x, y
                unit_len = bbox[2] - bbox[0]  # width of bbox
                unit_pts = [origin, origin, origin]  # init
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

                # Convert angles to radians
                pitch, yaw, roll = angles
                pitch = pitch * np.pi / 180
                yaw = -(yaw * np.pi / 180)
                roll = roll * np.pi / 180


                # X axis unit vector pointing to right in red
                unit_pts[0][0] += int(unit_len * (cos(yaw) * cos(roll)))
                unit_pts[0][1] += int(unit_len * (cos(pitch) * sin(roll) +
                                                  cos(roll) * sin(pitch) * sin(yaw)))

                # Y axis unit vector pointing to up in green
                unit_pts[1][0] += int(unit_len * (-cos(yaw) * sin(roll)))
                unit_pts[1][1] += int(unit_len * (cos(pitch) * cos(roll) -
                                                 sin(pitch) * sin(yaw) * sin(roll)))

                # Z axis unit vector pointing out of screen in blue
                unit_pts[2][0] += int(unit_len * (sin(yaw)))
                unit_pts[2][1] += int(unit_len * (-cos(yaw) * sin(pitch)))

                # Draw vectors
                for pt, c in zip(unit_pts, colors):
                    cv2.line(frame, origin, pt, color=c, thickness=3)

                # Add FPS text for debug
                if debug_mode:
                    cv2.putText(frame, f"FPS：{self.fps.fps():.2f}",
                               (frame.shape[1] - 30, frame.shape[0] - 10),
                               fontFace=cv2.FONT_HERSHEY_COMPLEX,
                               fontScale=0.45, color=(255, 0, 0))

                # Send annotated frame to be encoded to video
                img = dai.ImgFrame()
                img.setHeight(self.preview_size[0])
                img.setWidth(self.preview_size[1])
                img.setType(dai.RawImgFrame.Type.BGR888p)
                img.setFrame(frame.transpose([2, 0, 1]))  # channel first
                self.encoder.send(img)

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

    # TODO WIP
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
    sess = Session()

    # Register a graceful CTRL+C shutdown
    def shutdown(sig, frame):
        sess.running = False
    signal.signal(signal.SIGINT, shutdown)

    sess.run()
