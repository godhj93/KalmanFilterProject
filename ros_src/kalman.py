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


class MultiTargetTracker:

	def __init__(self):
		self.kalman_filter_done = False
		self.object_state = []
		self.iter_object_detection = 0
		# System matrix
		self.dt = 1.0/100.0
		
		self.A = np.matrix([
			[1,self.dt,0,0],
			[0,1,0,0],
			[0,0,1,self.dt],
			[0,0,0,1]],dtype=np.float32)
		
		self.H = np.matrix(
			[[1,0,0,0],
			[0,0,1,0]])

		# error covariance and other things
		self.Q = 1.0*np.eye(4)

		self.R = np.matrix(
			[[50,0],
			[0,50]])

		self.P = 100*np.eye(4)

		# initial guess state variable x = [x_pos,x_vel,y_pos,y_vel]'
		self.x = np.transpose(np.matrix([0,0,0,0]))
		
		self.pos_array = []
		# initialize ros
		rospy.init_node('kalman_filter', anonymous=True)
		self.yolo_sub = rospy.Subscriber("/yolo_output",yolo_coordinateArray,self.object_detection)
		self.image_sub = rospy.Subscriber("video/image",Image,self.image)


		# debug parameter
		self.bridge = CvBridge()
		self.i = 1
		self.i_max = 1
		self.firstRun = True


	def object_detection(self, data):

		
		# make sure the number of results
		self.object_state = []
		for ID,idx in enumerate(range(len(data.results))):

			self.object_state.append(
			  np.array([
			data.results[idx].x_center, #0
			data.results[idx].y_center, #1
			data.results[idx].w, # 2
			data.results[idx].h, #3
			data.results[idx].xmin, #4
			data.results[idx].xmax, #5
			data.results[idx].ymin, #6
			data.results[idx].ymax, #7
			data.results[idx].label+str(ID+1)
			] ,
			

			))


	def kalman_filter(self,data):
		

		if  len(data.results) > 0 :
			
		
			# idx = 0
			# self.pos_array = []

			# # Get estimated position with gaussian noise
			# self.x_pos_estimated = data.results[idx].x_center
			# self.y_pos_estimated = data.results[idx].y_center
			# self.label = data.results[idx].label + '1'

			
			# # Kalman filter 
			# print(self.x)
			# self.x_prediction = self.A * self.x

			# self.P_prediction = self.A * self.P * np.transpose(self.A) + self.Q

			# self.K = self.P_prediction * np.transpose(self.H) * np.linalg.inv(self.H * self.P_prediction * np.transpose(self.H) + self.R)

			# self.Z = np.transpose(np.matrix([self.x_pos_estimated, self.y_pos_estimated]))

			# self.x = self.x_prediction + self.K * (self.Z - self.H * self.x_prediction)

			# self.P = self.P_prediction - self.K * self.H * self.P_prediction

			# self.pos_array.append(self.x)
			# #self.pos_array.append((self.x_pos_estimated,self.y_pos_estimated,self.label))
						
			# #self.x = self.pos_array

			# self.firstRun = False


			
			

			self.pos_array = []
			for idx in range(len(data.results)):
				
				# Get estimated position with gaussian noise
				
				self.x_pos_estimated = data.results[idx].x_center
				self.y_pos_estimated = data.results[idx].y_center
				
				# Kalman filter 

				self.x_prediction = self.A * self.x

				self.P_prediction = self.A * self.P * np.transpose(self.A) + self.Q

				self.K = self.P_prediction * np.transpose(self.H) * np.linalg.inv(self.H * self.P_prediction * np.transpose(self.H) + self.R)

				self.Z = np.transpose(np.matrix([self.x_pos_estimated, self.y_pos_estimated]))

				self.x = self.x_prediction + self.K * (self.Z - self.H * self.x_prediction)

				self.P = self.P_prediction - self.K * self.H * self.P_prediction

				#self.pos_array.append(self.x)
				self.pos_array.append((self.x_pos_estimated,1,self.y_pos_estimated))
				
				self.firstRun = False



		else:
			pass
		
	def Euclidean(self,prev,current):

		for prev_id, prev_object in enumerate(prev):

			distance_list = []
			
			prev_x, prev_y = np.array(prev_object[0],dtype=np.int32), np.array(prev_object[1],dtype=np.int32)

			for current_id, current_object in enumerate(current):

				current_x, current_y = np.array(current_object[0], dtype=np.int32), np.array(current_object[1], dtype=np.int32)

				l2_distance = np.sqrt((current_x-prev_x)**2 + (current_y-prev_y)**2)

				distance_list.append(l2_distance)

			# rospy.loginfo(distance_list)
			# print('min', np.argmin(distance_list))
			current_id = np.argmin(distance_list)
			
			current[current_id][-1] = prev[prev_id][-1]



			




	def image(self,data):
		
		self.cv_rgb_image = self.bridge.imgmsg_to_cv2(data,'bgr8')
		for states in self.object_state:
			x_min = np.array(states[4],dtype=np.int32)
			y_min = np.array(states[6],dtype=np.int32)
			label = states[8]
		
			color = (0,150,255)#(np.random.randint(255),np.random.randint(255),np.random.randint(255))

			#self.cv_rgb_image = cv2.circle(self.cv_rgb_image, (pos_x,pos_y), 5, color, -1)
			cv2.putText(
				self.cv_rgb_image,
				label, 
				(x_min,y_min-10),
				 cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

		cv2.imshow('window',self.cv_rgb_image)

		cv2.waitKey(3)

		return self.cv_rgb_image

	





if __name__ =='__main__':


	
	MTT = MultiTargetTracker()
	rate = rospy.Rate(100)

	prev_objects = MTT.object_state

	while not rospy.is_shutdown():

		try:
			current_objects = MTT.object_state		

			MTT.Euclidean(prev_objects,current_objects)
				
			prev_objects = current_objects

		except:
			print("Erorr")


		rate.sleep()

	

	
	



		

	
