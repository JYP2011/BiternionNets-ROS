#!/usr/bin/env python
# encoding: utf-8

from os.path import abspath, expanduser, join as pjoin
import sys
from importlib import import_module

import numpy as np
import cv2

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

import DeepFried2 as df

from common import deg2bit, bit2deg, ensemble_biternions, subtractbg, cutout

from general_smoother.smoother import Smoother


# Distinguish between STRANDS and SPENCER.
try:
    from rwth_perception_people_msgs.msg import UpperBodyDetector
    from spencer_tracking_msgs.msg import TrackedPersons2d, TrackedPersons
except ImportError:
    from upper_body_detector.msg import UpperBodyDetector
    from mdl_people_tracker.msg import TrackedPersons2d, TrackedPersons


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
        self.counter = 0
        self.smoother_dict = dict()
        self.filter_method = 0

        modelname = rospy.get_param("~model", "head_50_50")
        weightsname = abspath(expanduser(rospy.get_param("~weights", ".")))
        rospy.loginfo("Predicting using {} & {}".format(modelname, weightsname))

        topic = rospy.get_param("~topic", "/biternion")
        self.pub = rospy.Publisher(topic, HeadOrientations, queue_size=3)
        self.pub_smooth = rospy.Publisher(topic + "_smoothed", HeadOrientations, queue_size=3)
        self.pub_vis = rospy.Publisher(topic + '/image', ROSImage, queue_size=3)
        self.pub_pa = rospy.Publisher(topic + "/pose", PoseArray, queue_size=3)
        self.pub_tracks = rospy.Publisher(topic + "/tracks", TrackedPersons, queue_size=3)

        # Ugly workaround for "jumps back in time" that the synchronizer sometime does.
        self.last_stamp = rospy.Time.now()

        # Create and load the network.
        netlib = import_module(modelname)
        self.net = netlib.mknet()
        self.net.__setstate__(np.load(weightsname))
        self.net.evaluate()

        self.aug = netlib.mkaug(None, None)
        self.preproc = netlib.preproc
        self.getrect = netlib.getrect

        # Do a fake forward-pass for precompilation.
        im = cutout(np.zeros((480,640,3), np.uint8), 0, 0, 150, 450)
        im = next(self.aug.augimg_pred(self.preproc(im), fast=True))
        self.net.forward(np.array([im]))
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
        subs.append(message_filters.Subscriber('/'.join(rgb.split('/')[:-1] + ['camera_info']), CameraInfo))

        tra3d = rospy.get_param("~tra3d", "")
        if src == "tra" and tra3d:
            subs.append(message_filters.Subscriber(tra3d, TrackedPersons))
            self.listener = TransformListener()

        ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=10, slop=10.0)
        ts.registerCallback(self.cb)

    def cb(self, src, rgb, d, caminfo, *more):
        # Ugly workaround because approximate sync sometimes jumps back in time.
        if rgb.header.stamp <= self.last_stamp:
            #rospy.logwarn("Jump back in time detected and dropped like it's hot")
            return

        self.last_stamp = rgb.header.stamp

        detrects = get_rects(src)
        t_ids = [(p2d.track_id) for p2d in src.boxes]

        # Early-exit to minimize CPU usage if possible.
        #if len(detrects) == 0:
        #    return


        # If nobody's listening, why should we be computing?
        if 0 == sum(p.get_num_connections() for p in (self.pub, self.pub_vis, self.pub_pa, self.pub_tracks)):
            return

        header = rgb.header
        bridge = CvBridge()
        rgb = bridge.imgmsg_to_cv2(rgb)[:,:,::-1]  # Need to do BGR-RGB conversion manually.
        d = bridge.imgmsg_to_cv2(d)
        imgs = []
        for detrect in detrects:
            detrect = self.getrect(*detrect)
            det_rgb = cutout(rgb, *detrect)
            det_d = cutout(d, *detrect)

            # Preprocess and stick into the minibatch.
            im = subtractbg(det_rgb, det_d, 1.0, 0.5)
            im = self.preproc(im)
            imgs.append(im)
            sys.stderr.write("\r{}".format(self.counter)) ; sys.stderr.flush()
            self.counter += 1

        # TODO: We could further optimize by putting all augmentations in a
        #       single batch and doing only one forward pass. Should be easy.
        if len(detrects):
            bits = [self.net.forward(batch) for batch in self.aug.augbatch_pred(np.array(imgs), fast=True)]
            preds = bit2deg(ensemble_biternions(bits)) - 90  # Subtract 90 to correct for "my weird" origin.
            # print(preds)
        else:
            preds = []
        
        # SMOOTHING
        fake_confs=[0.1] * len(preds) #TODO: de-fake
        new_angles = dict(zip(t_ids,list(zip(preds,fake_confs))))
        smoothed_angles = []
        smoothed_confs = []
        old_angles = []
        old_confs = []
        for t_id in t_ids:
            # 1) INIT ALL NEW
            if t_id not in self.smoother_dict:
                # new id, start new smoothing
                new_smoother = Smoother(deg2bit(new_angles[t_id][0]), new_angles[t_id][1], self.filter_method)
                smoothed_ang, smoothed_conf = new_angles[t_id]
                smoothed_angles.append(smoothed_ang)
                smoothed_confs.append(smoothed_conf)
                self.smoother_dict[t_id] = new_smoother
                continue
            old_ang, old_conf = self.smoother_dict[t_id].getCurrentValueAndConfidence()
            old_ang = bit2deg(np.array([old_ang]))
            old_angles.append(old_ang)
            old_confs.append(old_conf)
            # 2) PREDICT ALL EXISTING
            self.smoother_dict[t_id].predict()
            # 3) UPDATE ALL EXISTING
            self.smoother_dict[t_id].update(deg2bit(new_angles[t_id][0]), new_angles[t_id][1])
            # 4) DELETE ALL OLD
            # TODO: deletion logic, or just keep all in case they return and predict up to then

            # append result here to keep id_idx
            smoothed_ang, smoothed_conf = self.smoother_dict[t_id].getCurrentValueAndConfidence()
            smoothed_ang = bit2deg(np.array([smoothed_ang]))
            smoothed_angles.append(smoothed_ang)
            smoothed_confs.append(smoothed_conf)

        # published unsmoothed
        if 0 < self.pub.get_num_connections():
            self.pub.publish(HeadOrientations(
                header=header,
                angles=list(preds),
                confidences=[0.83] * len(imgs),
                ids=list(t_ids)
            ))

        # publish smoothed
        if 0 < self.pub_smooth.get_num_connections():
            self.pub_smooth.publish(HeadOrientations(
                header=header,
                angles=list(smoothed_angles),
                confidences=list(smoothed_confs),
                ids = list(t_ids)
            ))

        # Visualization
        if 0 < self.pub_vis.get_num_connections():
            rgb_vis = rgb[:,:,::-1].copy()
            for detrect, alpha, beta, gamma in zip(detrects, preds, old_angles, smoothed_angles):
                l, t, w, h = self.getrect(*detrect)
                px =  int(round(np.cos(np.deg2rad(alpha))*w/2))
                py = -int(round(np.sin(np.deg2rad(alpha))*h/2))
                px_old =  int(round(np.cos(np.deg2rad(beta))*w/2))
                py_old = -int(round(np.sin(np.deg2rad(beta))*h/2))
                px_smooth =  int(round(np.cos(np.deg2rad(gamma))*w/2))
                py_smooth = -int(round(np.sin(np.deg2rad(gamma))*h/2))
                cv2.rectangle(rgb_vis, (detrect[0], detrect[1]), (detrect[0]+detrect[2],detrect[1]+detrect[3]), (0,255,255), 1)
                cv2.rectangle(rgb_vis, (l,t), (l+w,t+h), (0,255,0), 2)
                cv2.line(rgb_vis, (l+w//2, t+h//2), (l+w//2+px_smooth,t+h//2+py_smooth), (0,255,0), 2)
                cv2.line(rgb_vis, (l+w//2, t+h//2), (l+w//2+px,t+h//2+py), (255,0,0), 1)
                cv2.line(rgb_vis, (l+w//2, t+h//2), (l+w//2+px_old,t+h//2+py_old), (0,0,255), 1)
                # cv2.putText(rgb_vis, "{:.1f}".format(alpha), (l, t+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
            vismsg = bridge.cv2_to_imgmsg(rgb_vis, encoding='rgb8')
            vismsg.header = header  # TODO: Seems not to work!
            self.pub_vis.publish(vismsg)

        if 0 < self.pub_pa.get_num_connections():
            fx, cx = caminfo.K[0], caminfo.K[2]
            fy, cy = caminfo.K[4], caminfo.K[5]

            poseArray = PoseArray(header=header)

            for (dx, dy, dw, dh, dd), alpha in zip(get_rects(src, with_depth=True), preds):
                dx, dy, dw, dh = self.getrect(dx, dy, dw, dh)

                # PoseArray message for boundingbox centres
                poseArray.poses.append(Pose(
                    position=Point(
                        x=dd*((dx+dw/2.0-cx)/fx),
                        y=dd*((dy+dh/2.0-cy)/fy),
                        z=dd
                    ),
                    # TODO: Use global UP vector (0,0,1) and transform into frame used by this message.
                    orientation=Quaternion(*quaternion_about_axis(np.deg2rad(alpha), [0, -1, 0]))
                ))

            self.pub_pa.publish(poseArray)

        if len(more) == 1 and 0 < self.pub_tracks.get_num_connections():
            t3d = more[0]
            try:
                self.listener.waitForTransform(header.frame_id, t3d.header.frame_id, rospy.Time.now(), rospy.Duration(1.0))
                for track, alpha in zip(t3d.tracks, preds):
                    track.pose.pose.orientation = self.listener.transformQuaternion(t3d.header.frame_id, QuaternionStamped(
                        header=header,
                        # TODO: Same as above!
                        quaternion=Quaternion(*quaternion_about_axis(np.deg2rad(alpha), [0, -1, 0]))
                    )).quaternion
                self.pub_tracks.publish(t3d)
            except TFException:
                print("TFException")
                pass


if __name__ == "__main__":
    rospy.init_node("biternion_predict")

    # Add the "models" directory to the path!
    sys.path.append(pjoin(RosPack().get_path('biternion'), 'scripts'))
    sys.path.append(pjoin(RosPack().get_path('biternion'), 'models'))

    p = Predictor()
    rospy.spin()
    rospy.loginfo("Predicted a total of {} UBDs.".format(p.counter))
