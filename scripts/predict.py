#!/usr/bin/env python
# encoding: utf-8

from os.path import abspath, expanduser, join as pjoin
import os, sys
from importlib import import_module
from collections import defaultdict

import numpy as np
import cv2
import time

import rospy
from tf import TransformListener, Exception as TFException
from tf.transformations import quaternion_about_axis
from rospkg import RosPack
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image as ROSImage, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, QuaternionStamped
from biternion.msg import HeadOrientations
from visualization_msgs.msg import Marker
from common import deg2bit, bit2deg
from general_smoother.smoother2 import *

# Distinguish between STRANDS and SPENCER.
try:
    from rwth_perception_people_msgs.msg import UpperBodyDetector
    from spencer_tracking_msgs.msg import TrackedPersons2d
except ImportError:
    from upper_body_detector.msg import UpperBodyDetector
    from mdl_people_tracker.msg import TrackedPersons2d


def get_rects(msg, with_depth=False):
    if isinstance(msg, TrackedPersons2d):
        return [(p2d.x, p2d.y, p2d.w, p2d.h) + ((p2d.depth,) if with_depth else tuple()) for p2d in msg.boxes]
    elif isinstance(msg, UpperBodyDetector):
        return list(zip(*([msg.pos_x, msg.pos_y, msg.width, msg.height] + ([msg.median_depth] if with_depth else []))))
    else:
        raise TypeError("Unknown source type: {}".format(type(msg)))

class Predictor(object):
    def __init__(self):
        rospy.loginfo("HE-MAN ready!")

        self.result_path = rospy.get_param("~result_path", "/home/stefan/results/spencer_tracker/anno_seq_03-e1920-2505_results/last_run/") #TODO: de-hc
        self.do_save_results = os.path.isdir(self.result_path)
        if self.do_save_results:
            self.results_file_smoothed = open(pjoin(self.result_path,"heads_smoothed.txt"),'w')
            self.results_file_orig = open(pjoin(self.result_path,"heads_orig.txt"),'w')
            self.results_file_smoothed.close()
            self.results_file_orig.close()

        self.seen_counter = 0
        self.ana_counter = 0
        filtermaker = rospy.get_param("~filter", "GHFilter(g=0.5)")
        self.smoother_dict = defaultdict(lambda: eval(filtermaker)) if rospy.get_param("~smooth", False) else None
        self.stride = rospy.get_param("~stride", 1)

        # for timing
        self.cycle_idx = 0
        self.hz_curr = 0.0
        self.hz_sum = 0.0
        self.hz_mean = 0.0

        modelname = rospy.get_param("~model", "head_50_50")
        weightsname = abspath(expanduser(rospy.get_param("~weights", ".")))
        rospy.loginfo("Predicting using {} & {}".format(modelname, weightsname))

        topic = rospy.get_param("~topic", "/biternion")
        self.pub = rospy.Publisher(topic, HeadOrientations, queue_size=3)
        self.pub_smooth = rospy.Publisher(topic + "_smoothed", HeadOrientations, queue_size=3)
        self.pub_vis = rospy.Publisher(topic + '/image', ROSImage, queue_size=3)
        self.pub_pa = rospy.Publisher(topic + "/pose", PoseArray, queue_size=3)

        # Ugly workaround for "jumps back in time" that the synchronizer sometime does.
        self.last_stamp = rospy.Time.now()
        self.time_travels = 0

        # Create and load the network.
        netlib = import_module(modelname)
        self.model = netlib.Model(weightsname, GPU=False)

        # Do a fake forward-pass for precompilation/GPU init/...
        self.model(np.zeros((480,640,3), np.uint8),
                   np.zeros((480,640), np.float32), [(0,0,150,450)])
        rospy.loginfo("BiternionNet initialized")

        src = rospy.get_param("~src", "tra")
        subs = []
        if src == "tra":
            subs.append(message_filters.Subscriber(rospy.get_param("~tra", "/TODO"), TrackedPersons2d))
        elif src == "ubd":
            subs.append(message_filters.Subscriber(rospy.get_param("~ubd", "/upper_body_detector/detections"), UpperBodyDetector))
        else:
            raise ValueError("Unknown source type: " + src)

        rgb = rospy.get_param("~rgb", "/head_xtion/rgb/image_rect_color")
        subs.append(message_filters.Subscriber(rgb, ROSImage))
        subs.append(message_filters.Subscriber(rospy.get_param("~d", "/head_xtion/depth/image_rect_meters"), ROSImage))

        syncSlop = rospy.get_param("~sync_slop", 0.05)
        syncQueueSize = rospy.get_param("~sync_queue_size", 3)
        ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=syncQueueSize, slop=syncSlop)

        ts.registerCallback(self.cb)

    def cb(self, src, rgb, d):
        dt = (rgb.header.stamp - self.last_stamp).to_sec()
        self.last_stamp = rgb.header.stamp

        # Ugly workaround because approximate sync sometimes jumps back in time.
        if dt <= 0:
            rospy.logwarn("Jump back in time detected and dropped like it's hot")
            self.time_travels += 1

        # TIMING START
        self.cycle_idx += 1
        cycle_start_time = time.time()

        detrects = get_rects(src)
        t_ids = [(p2d.track_id) for p2d in src.boxes]
        self.seen_counter += len(t_ids)

        header = rgb.header
        bridge = CvBridge()
        rgb = bridge.imgmsg_to_cv2(rgb, desired_encoding='rgb8')
        d = bridge.imgmsg_to_cv2(d)

        is_freeflight_cycle = (self.cycle_idx % self.stride) != 0
        if not is_freeflight_cycle:
            # Do the extraction and prediction
            preds_deg, preds_confs = self.model(rgb, d, detrects)
            preds_bit = deg2bit(preds_deg)
            self.ana_counter += len(preds_bit)
        else:
            preds_bit = [None] * len(t_ids)  # Note: this will take care of correct filtering below.

        # SMOOTHING
        smoothed_biternions = {}
        smoothed_confs = {}
        old_biternions = {}
        old_confs = {}
        if self.smoother_dict is not None:
            for t_id, biternion in zip(t_ids, preds_bit):
                if is_freeflight_cycle and t_id not in self.smoother_dict:
                    continue   # Don't start a new smoother when I don't have a value.
                if t_id in self.smoother_dict:  # For visualization only, get the previous smooth value.
                    insert_each(t_id, self.smoother_dict[t_id].get(), old_biternions, old_confs)
                smoother = self.smoother_dict[t_id]
                smoother.tick(dt, biternion, conf=None)
                insert_each(t_id, smoother.get(), smoothed_biternions, smoothed_confs)

            # publish smoothed
            if 0 < self.pub_smooth.get_num_connections():
                self.pub_smooth.publish(HeadOrientations(
                    header=header,
                    angles=[bit2deg(a) for a in smoothed_angles],
                    confidences=list(smoothed_confs),
                    ids = list(t_ids)
                ))

        # published unsmoothed
        if 0 < self.pub.get_num_connections():
            self.pub.publish(HeadOrientations(
                header=header,
                angles=list(preds_deg) if not is_freeflight_cycle else [],
                confidences=list(preds_confs) if not is_freeflight_cycle else [],
                ids=list(t_ids)
            ))

        # TIMING END
        cycle_end_time = time.time()
        cycle_duration = cycle_end_time - cycle_start_time
        self.hz_curr = 1./(cycle_duration)
        self.hz_sum = self.hz_sum + self.hz_curr
        self.hz_mean = self.hz_sum/self.cycle_idx

        if self.do_save_results:
            with open(self.results_file_smoothed.name,'a') as f_s:
                for t_id in t_ids:
                    if t_id in smoothed_biternions:
                        f_s.write("{0:d} {1:d} {2:.2f} {3:.2f}\n".format(src.frame_idx, t_id, bit2deg(smoothed_biternions[t_id]), smoothed_confs[t_id]))

            if not is_freeflight_cycle:
                with open(self.results_file_orig.name,'a') as f_o:
                    for t_id, bit, conf in zip(t_ids, preds_bit, preds_confs):
                        f_o.write("{0:d} {1:d} {2:.2f} {3:.2f}\n".format(src.frame_idx, t_id, bit2deg(bit), conf))

        # Visualization
        # TODO: Visualize confidence, too.
        if 0 < self.pub_vis.get_num_connections():
            rgb_vis = rgb.copy()
            for detrect, t_id, alpha in zip(detrects, t_ids, preds_bit):
                l, t, w, h = self.model.getrect(*detrect)
                cv2.rectangle(rgb_vis, (detrect[0], detrect[1]), (detrect[0]+detrect[2],detrect[1]+detrect[3]), (0,255,255), 1)
                cv2.rectangle(rgb_vis, (l,t), (l+w,t+h), (0,255,0), 2)

                if t_id in smoothed_biternions:
                    gamma = bit2deg(smoothed_biternions[t_id])
                    px_smooth =  int(round(np.cos(np.deg2rad(gamma))*w/2))
                    py_smooth = -int(round(np.sin(np.deg2rad(gamma))*h/2))
                    cv2.line(rgb_vis, (l+w//2, t+h//2), (l+w//2+px_smooth,t+h//2+py_smooth), (0,255,0), 2)
                if alpha is not None:
                    px =  int(round(np.cos(np.deg2rad(bit2deg(alpha)))*w/2))
                    py = -int(round(np.sin(np.deg2rad(bit2deg(alpha)))*h/2))
                    cv2.line(rgb_vis, (l+w//2, t+h//2), (l+w//2+px,t+h//2+py), (255,0,0), 1)
                if t_id in old_biternions:
                    beta = bit2deg(old_biternions[t_id])
                    px_old =  int(round(np.cos(np.deg2rad(beta))*w/2))
                    py_old = -int(round(np.sin(np.deg2rad(beta))*h/2))
                    cv2.line(rgb_vis, (l+w//2, t+h//2), (l+w//2+px_old,t+h//2+py_old), (0,0,255), 1)
                # cv2.putText(rgb_vis, "{:.1f}".format(bit2deg(alpha)), (l, t+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
            vismsg = bridge.cv2_to_imgmsg(rgb_vis, encoding='rgb8')
            vismsg.header = header  # TODO: Seems not to work!
            self.pub_vis.publish(vismsg)


if __name__ == "__main__":
    rospy.init_node("biternion_predict")

    # Add the "models" directory to the path!
    sys.path.append(pjoin(RosPack().get_path('biternion'), 'scripts'))
    sys.path.append(pjoin(RosPack().get_path('biternion'), 'models'))

    p = Predictor()
    rospy.spin()
    rospy.loginfo("Analyzed a total of {} heads, out of {} seen heads.".format(p.ana_counter, p.seen_counter))
    rospy.loginfo("Mean HZ: {}".format(p.hz_mean))
    rospy.loginfo("Time travel problems: {}".format(p.time_travels))
