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


class KalmanBoxTracker:

	_counter = 0

	def __init__(self):
		# Kalman Filter 
		KalmanBoxTracker._counter +=1

		self.id = KalmanBoxTracker._counter

		self.hit = 0
		#print("Created new tracker, ID : ",self.id)

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
		self.kf.P[4:,4:] *= 10
		self.kf.P *= 100.
		self.kf.Q[-1,-1] *= 0.1
		self.kf.Q[4:,4:] *= 0.1

	
		

	def save_z(self,u,v,s,r):
		self.u = u
		self.v = v
		self.s = s
		self.r = r

	def load_z(self):
		return self.u,self.v,self.s,self.r

	def prediction(self, u,v,s,r):
		
		z = [u,v,s,r]
		self.kf.predict()
		self.kf.update(z)
		

		# Kalman filter prediction

		uu = self.kf.x[0]
		vv = self.kf.x[1]
		ss = self.kf.x[2]
		rr = self.kf.x[3]
		if ss < 0:
			ss = 0
		elif rr < 0:
			rr = 0

		return uu,vv,ss,rr#,ID



class Tracker:

	def __init__(self):

		rospy.init_node('tracker', anonymous = True)
		self.yolo_sub = rospy.Subscriber("/yolo_output",yolo_coordinateArray, self.object_detection)
		self.image_sub = rospy.Subscriber("video/image",Image,self.get_image)
		self.bridge = CvBridge()

		self.object_state = []
		self.kalman_boxes = []
		self.kalman_tracks = []
		self.kalman_tracks_new = []
		self.firstRun = 0

		self.IDnum = 0

	def object_detection(self,data):

		# Get object information
		self.object_state = []
		for ID, idx in enumerate(range(len(data.results))):

			u = data.results[idx].x_center
			v = data.results[idx].y_center
			s = data.results[idx].w * data.results[idx].h
			r = float(data.results[idx].w) / float(data.results[idx].h)
			
			self.object_state.append(
			[
				np.array([u,v,s,r,self.IDnum])	
			])

			self.IDnum += 1

			#self.draw(u,v,s,r,'yolo', 0,255,0)



		if self.firstRun == 0 :

			self.firstRun +=1 
			# Kalman filter

			for state in self.object_state:
				
				u,v,s,r,ID = state[0][:5]
				
				tracker = KalmanBoxTracker()
				
				uu,vv,ss,rr = tracker.prediction(u,v,s,r)

				tracker.save_z(uu,vv,ss,rr)
				
				self.kalman_tracks.append(tracker)


		else:
			print('====')
			print(len(self.kalman_tracks))


			# Keep going to prediction
			for tracker in self.kalman_tracks:
				
				#u,v,s,r = tracker.kf.x[:4]
				u,v,s,r = tracker.load_z()
				tracker.prediction(u,v,s,r)
				#self.draw(u,v,s,r,'kalman',255,255,255)
				#self.draw(u,v,s,r,'kalman', 0,255,255)

			measurement_list = []
			for state in self.object_state:

				measurement_list.append(state[0][:4])


				# tracker_new = KalmanBoxTracker()
				# u,v,s,r=state[0][:4]

				# tracker_new.save_z(u,v,s,r)
				# #tracker_new.prediction(u,v,s,r) 

				# #tracker_new.save_z(u,v,s,r)

				# self.kalman_tracks_new.append(tracker_new)

			

			# #IOU matching
			iou_table = np.zeros(shape=(len(self.kalman_tracks), len(measurement_list)))
			#print(iou_table.shape)

			for idx_measurement,measurement in enumerate(measurement_list):

				for idx_kalman,tracker in enumerate(self.kalman_tracks):
					
					u_tracker ,v_tracker ,s_tracker, r_tracker = tracker.load_z()

					u_measurement, v_measurement, s_measurement, r_measurement = measurement

					self.draw(u_tracker ,v_tracker ,s_tracker, r_tracker,tracker.id,0,255,0)
					self.draw(u_measurement, v_measurement, s_measurement, r_measurement,'measure',255,0,0)


					
					iou_table[idx_kalman][idx_measurement] = self.iou(
												[u_tracker ,v_tracker ,s_tracker, r_tracker],
												[u_measurement, v_measurement, s_measurement, r_measurement]
												)
			#print('iou_table',iou_table)

			row_ind, col_ind = linear_sum_assignment(-iou_table)

			optimal_assignment = iou_table[row_ind,col_ind]
			
			assigned_col = []
			assigned_row = []

			
			for idx in range(len(optimal_assignment)):
				
				row, col = np.where(iou_table == optimal_assignment[idx])
				assigned_row.append(row[0])
				assigned_row.append(col[0])
				u_measurement = measurement_list[col[0]][0]
				v_measurement = measurement_list[col[0]][1]
				s_measurement = measurement_list[col[0]][2]
				r_measurement = measurement_list[col[0]][3]

				self.kalman_tracks[row[0]].save_z(u_measurement, v_measurement, s_measurement, r_measurement)

			 	
		
			iou_row_list = np.arange(iou_table.shape[0])
			iou_col_list = np.arange(iou_table.shape[1])

			
			for row in iou_row_list:
				if row not in assigned_row:
					print('@@@@@@@@@@@@@@@@',row)
					self.kalman_tracks[row].hit += 1
					print('hit',self.kalman_tracks[row].hit)
					# if self.kalman_tracks[row].hit >= 3:
					# 	self.kalman_tracks_new.append(self.kalman_tracks[row])
					# print(self.kalman_tracks[row].id,self.kalman_tracks[row].hit)
				else:
					pass#self.kalman_tracks[row].hit += 1
				# elif row in assigned_row:
				# 	self.kalman_tracks[row].hit += 1

				if self.kalman_tracks[row].hit >= 50 :
					print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
					self.kalman_tracks.remove(self.kalman_tracks[row])
				elif self.kalman_tracks[row].hit <= 0 :
					pass

			for col in iou_col_list:

				u_measurement = measurement_list[col][0]
				v_measurement = measurement_list[col][1]
				s_measurement = measurement_list[col][2]
				r_measurement = measurement_list[col][3]

				if col not in assigned_row:
					print('new detection',col)
					tracker = KalmanBoxTracker()
					tracker.save_z(u_measurement,v_measurement,s_measurement,r_measurement)

					self.kalman_tracks.append(tracker)

					

			# 	# for idx in range(len(optimal_assignment)):



			# 	# 	# row,col = np.where(iou_table == optimal_assignment[idx])
			# 	# 	# assigned_row.append(row[0])
			# 	# 	# assigned_col.append(col[0])
			# 	# 	# u,v,s,r = self.kalman_tracks[row[0]].load_z()

			# 	# 	# uu,vv,ss,rr = self.kalman_tracks[row[0]].prediction(u,v,s,r)

			# 	# 	self.kalman_tracks_new.append(self.kalman_tracks[row[0]])
			# 	# 	self.draw(uu,vv,ss,rr,'',0,255,255)





			# # iou_table_row_list = np.arange(iou_table.shape[0])
			# # iou_table_col_list = np.arange(iou_table.shape[1])

			# #print(iou_table_row_list)
			# # if len(assigned_row) > 0 :
			# # 	for item_row in iou_table_row_list:
			# # 		print(item_row,assigned_row)
			# # 		if item_row not in assigned_row:
			# # 			print('gd')
			# # if len(assigned_col) > 0 : 
			# # 	for item_col in iou_table_col_list:
			# # 		if item_col not in assigned_col:
						
			# # 			u,v,s,r = self.kalman_tracks_new[item_col].load_z()
			# # 			self.kalman_tracks_new[item_col].prediction(u,v,s,r)

			# self.kalman_tracks = self.kalman_tracks_new

			# self.kalman_tracks_new = []



			


				# self.object_state[col[0]][0][-1] = kalman_id

				# self.kalman_tracks[col[0]] = self.kalman_tracks[int(kalman_id)]

				# print(len(self.kalman_tracks),self.kalman_tracks)

			# After matching, clear list
			# self.kalman_boxes = []
		
		# for i,state in enumerate(self.object_state):

		# 	tracker = KalmanBoxTracker()
		# 	self.kalman_tracks.append(tracker)
 
		

		# tracker_list = []
		# for tracker,state in zip(self.kalman_tracks,self.object_state):
		# 	u,v,s,r,ID = state[0][0],state[0][1],state[0][2],state[0][3],state[0][4]

		# 	uu, vv, ss, rr , kalman_id = tracker.prediction(state)
		# 	tracker_list.append(tracker)
		# 	self.kalman_boxes.append(np.array([uu, vv, ss, rr , ID]))
		# 	self.draw(u,v,s,r,ID,255,0,0)
		# 	self.draw(uu,vv,ss,rr,kalman_id,0,255,0)
		
		
		
		cv2.imshow('window', self.cv_rgb_image)
		cv2.waitKey(3)	

			

	




	def iou(self,box_a,box_b):
		
		
		xmin_a, ymin_a, xmax_a, ymax_a = self.get_box_point(box_a[0],box_a[1],box_a[2],box_a[3])
		xmin_b, ymin_b, xmax_b, ymax_b = self.get_box_point(box_b[0],box_b[1],box_b[2],box_b[3])


		box_a_area = np.abs(xmax_a-xmin_a) * np.abs(ymax_a-ymin_a)
		box_b_area = np.abs(xmax_b-xmin_b) * np.abs(ymax_b-ymin_b)
		
		
		iou_xmin = np.max([xmin_a,xmin_b])
		iou_ymin = np.max([ymin_a,ymin_b])

		iou_xmax = np.min([xmax_a,xmax_b])
		iou_ymax = np.min([ymax_a,ymax_b])

		iou_width = np.abs(iou_xmax-iou_xmin)
		iou_height = np.abs(iou_ymax-iou_ymin)

		overlapping_region = iou_width * iou_height
		combined_region = box_a_area + box_b_area - overlapping_region

		#self.cv_rgb_image = cv2.rectangle(self.cv_rgb_image, (int(iou_xmin),int(iou_ymin)), (int(iou_xmax),int(iou_ymax)), (255,255,255),-1)
		
		
		return float(overlapping_region)/float(combined_region)
		



	def get_box_point(self,u,v,s,r):

		w = np.sqrt(s*r)

		h = np.sqrt(s/r)
		
		xmin = u - w/2.
		xmax = u + w/2.
		ymin = v - h/2.
		ymax = v + h/2.

		return xmin,ymin,xmax,ymax



	def draw(self, uu,vv,ss,rr,ID,blue,green,red):

		xmin, ymin, xmax, ymax = self.get_box_point(uu,vv,ss,rr)		
		self.cv_rgb_image = cv2.rectangle(self.cv_rgb_image, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (blue,green,red))

		cv2.putText(
				self.cv_rgb_image,
				str(ID), 
				(int(xmin),int(ymin)-10),
				 cv2.FONT_HERSHEY_SIMPLEX, 1, (blue,green,red), 2, cv2.LINE_AA)

		
		

	def get_image(self, data):

		self.cv_rgb_image = self.bridge.imgmsg_to_cv2(data,'bgr8')

	
	def kalman(self, states):

		
		self.kf.P[4:,4:] *= 10.
		self.kf.P *= 10.
		u = states[0][0]
		v = states[0][1]
		s = states[0][2]
		r = states[0][3]
		ID = states[0][4]

	
		x = [u,v,s,r]
		self.kf.predict(x,)
		self.kf.update([u,v,s,r])
		

		# Kalman filter prediction

		uu = self.kf.x[0]
		vv = self.kf.x[1]
		ss = self.kf.x[2]
		rr = self.kf.x[3]
		if ss < 0:
			ss = 0
		elif rr < 0:
			rr = 0

		return uu,vv,ss,rr,ID
		












if __name__=='__main__':

	Tracker()

	rospy.spin()

	