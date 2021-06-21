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
from scipy.optimize import linear_sum_assignment



def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class Tracker:

	def __init__(self):

		rospy.init_node('tracker', anonymous = True)
		self.yolo_sub = rospy.Subscriber("/yolo_output",yolo_coordinateArray, self.object_detection)
		self.image_sub = rospy.Subscriber("video/image",Image,self.get_image)
		self.bridge = CvBridge()

		self.kalman_history = [] 
		self.object_state = []
		self.kalman_boxes = []
		self.firstRun = 0

		self.IDnum = 0

		# Kalman Filter 

		self.kf = KalmanFilter(dim_x=7, dim_z=4)

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


		self.kf.R[2:,2:] *= 10.
		self.kf.P[4:,4:] *= 1000.
		self.kf.P *= 10.
		self.kf.Q[-1,-1] *= 0.01
		self.kf.Q[4:,4:] *= 0.01

		self.kf.x[:4] = convert_bbox_to_z(bbox)

		self.time_since_update = 0
		self.id = KalmanBoxTracker.count
		KalmanBoxTracker.count += 1
		self.history = []
		self.hits = 0
		self.hit_streak = 0
		self.age = 0

	def update(self,bbox):

		self.time_since_update = 0
		self.history = []
		self.hits += 1
		self.hit_streak += 1
		self.kf.update(convert_bbox_to_z(bbox))

	def predict(self):

		if((self.kf.x[6]+self.kf.x[2])<=0):
			self.kf,x[6] *= 0.

		self.kf.predict()
		self.age += 1

		if(self.time_since_update > 0):
			self.hit_streak = 0

		self.time_since_update+=1
		self.history.append(convert_x_to_bbox(self.kf.x))

		return self.history[-1]

	def get_state(self):

		return convert_x_to_bbox(self.kf.x)

	def object_detection(self,data):

		self.object_state = []
		for ID, idx in enumerate(range(len(data.results))):

			u = data.results[idx].x_center
			v = data.results[idx].y_center
			s = data.results[idx].w * data.results[idx].h
			r = float(data.results[idx].w) / float(data.results[idx].h)
			confidence = data.results[idx].confidence
			self.object_state.append(
			[
				[u,v,s,r,self.IDnum,confidence]
			])

			self.IDnum += 1


		for states in self.object_state:
			uu,vv,ss,rr, kalman_ID =self.kalman(states)

			self.draw(uu,vv,ss,rr,kalman_ID,255,255,255)

		cv2.imshow('window', self.cv_rgb_image)
		cv2.waitKey(3)




	def get_image(self, data):

		self.cv_rgb_image = self.bridge.imgmsg_to_cv2(data,'bgr8')

	

	def kalman(self, states):

		print('kalstates',states)

		
		u = states[0][0]
		v = states[0][1]
		s = states[0][2]
		r = states[0][3]
		ID = states[0][4]

	
		
		self.kf.predict()
		self.kf.update([u,v,s,r])
		

		# Kalman filter prediction
		print(u,v,s,r)
		print([item for item in self.kf.x])
		uu = self.kf.x[0]
		vv = self.kf.x[1]
		ss = self.kf.x[2]
		rr = self.kf.x[3]
		if ss < 0:
			ss = 0
			
		elif rr < 0:
			rr = 0

		#return uu,vv,ss,rr,ID
		return uu,vv,ss,rr,ID


	def draw(self, uu,vv,ss,rr,ID,blue,green,red):

		xmin, ymin, xmax, ymax = self.get_box_point(uu,vv,ss,rr)		
		self.cv_rgb_image = cv2.rectangle(self.cv_rgb_image, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (blue,green,red))

		cv2.putText(
				self.cv_rgb_image,
				str(ID), 
				(int(xmin),int(ymin)-10),
				 cv2.FONT_HERSHEY_SIMPLEX, 1, (blue,green,red), 2, cv2.LINE_AA)

		


	def get_box_point(self,u,v,s,r):

		w = np.sqrt(s*r)

		h = np.sqrt(s/r)
		
		xmin = u - w/2.
		xmax = u + w/2.
		ymin = v - h/2.
		ymax = v + h/2.

		return xmin,ymin,xmax,ymax


if __name__=='__main__':

	Tracker()
	rate = rospy.Rate(20)
	rospy.spin()

	