#!/usr/bin/env python


import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ros_tkdnn.msg import yolo_coordinateArray
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import copy
import time
from filterpy.kalman import KalmanFilter

class Tracker:

	def __init__(self):

		rospy.init_node('tracker', anonymous = True)
		self.yolo_sub = rospy.Subscriber("/yolo_output",yolo_coordinateArray, self.object_detection)
		self.image_sub = rospy.Subscriber("video/image",Image,self.draw)
		self.bridge = CvBridge()

		self.object_state = []

		# Kalman Filter 

		self.kf = KalmanFilter(dim_x=7, dim_z=4)

		self.kf.x = np.array([0,0,0,0,0,0,0])

		self.kf.F = np.array([
			[1,0,0,0,1,0,0],
			[0,1,0,0,0,1,0],
			[0,0,1,0,0,0,1],
			[0,0,0,1,0,0,0],
			[0,0,0,0,1,0,0],
			[0,0,0,0,0,1,0],
			[0,0,0,0,0,0,1]
			])

		self.kf.H = np.array([
			[1,0,0,0,0,0,0],
			[0,1,0,0,0,0,0],
			[0,0,1,0,0,0,0],
			[0,0,0,1,0,0,0]
			])


		self.kf.R[2:,2:] *= 10
		self.kf.P[4:,4:] *= 100
		#self.kf.P *= 10.
		self.kf.Q[-1,-1] *= 0.1
		self.kf.Q[4:,4:] *= 0.1





	def object_detection(self,data):

		self.object_state = []
		for ID,idx in enumerate(range(len(data.results))):

			u=data.results[idx].x_center
			v=data.results[idx].y_center
			s=data.results[idx].w * data.results[idx].h
			r=float(data.results[idx].w) / float(data.results[idx].h)

			self.object_state.append(
			[
				np.array([u,v,s,r])	
			])

		# print("Detected object :", len(self.object_state))
		# for item in self.object_state:
		# 	print(item)

	
	def draw(self, data):

		self.cv_rgb_image = self.bridge.imgmsg_to_cv2(data,'bgr8')
		# cv2.imshow('window', self.cv_rgb_image)
		# cv2.waitKey(3)

	def kalman(self, states):

		
		u = states[0][0]
		v = states[0][1]
		s = states[0][2]
		r = states[0][3]

		print("==")
		
		self.kf.predict()
		self.kf.update([u,v,s,r])
		

		# Kalman filter prediction

		uu = self.kf.x[0]
		vv = self.kf.x[1]
		ss = self.kf.x[2]
		rr = self.kf.x[3] 

		w = np.sqrt(ss*rr)

		h = np.sqrt(ss/rr)
		
		xmin = uu - w/2.
		xmax = uu + w/2.
		ymin = vv - h/2.
		ymax = vv + h/2.

		#self.cv_rgb_image = cv2.circle(self.cv_rgb_image, (int(u),int(v)), 5 , (0,255,0), -1)


		self.cv_rgb_image = cv2.rectangle(self.cv_rgb_image, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,0,255))
		#self.cv_rgb_image = cv2.circle(self.cv_rgb_image, (int(self.kf.x[0]),int(self.kf.x[1])), 5 , (0,0,255), -1)
		cv2.imshow('window', self.cv_rgb_image)
		cv2.waitKey(3)












if __name__=='__main__':

	MTT = Tracker()

	
	rate = rospy.Rate(100)

	while not rospy.is_shutdown():


		rate.sleep()

		for states in MTT.object_state:
			MTT.kalman(states)
