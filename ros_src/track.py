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

class box:
    
    def __init__(self,x0,y0,x1,y1):
        self.xmin = x0
        self.ymin = y0
        self.xmax = x1
        self.ymax = y1    


class KalmanBoxTracker:

	_counter = 0
	
	def __init__(self):
		# Kalman Filter 
		self.age = 30
		KalmanBoxTracker._counter +=1

		self.id = KalmanBoxTracker._counter

		self.color = (np.random.randint(255),np.random.randint(255),np.random.randint(255))

		self.hit = self.age
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

		# self.kf.R[2:,2:] *= 10
		# self.kf.P[4:,4:] *= 10
		# self.kf.P *= 100.
		# self.kf.Q[-1,-1] *= 0.1
		# self.kf.Q[4:,4:] *= 0.1

		self.kf.R[0,0] = 100
		self.kf.R[1,1] = 100
		self.kf.R[2,2] = 10
		self.kf.R[3,3] = 10
		#self.kf.P[4:,4:] *= 100
		self.kf.P *= 1000
		self.kf.Q[0,0] = 10
		self.kf.Q[1,1] = 10
		self.kf.Q[2,2] = 10
		self.kf.Q[3,3] = 10

		

	
		

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

	


		if self.firstRun == 0 :
			print("first number of tracker",len(self.kalman_tracks))
			
			if self.object_state:
				self.firstRun +=1 
				# Kalman filter

				for state in self.object_state:
					
					u,v,s,r = state[0][:4]
					
					tracker = KalmanBoxTracker()

					tracker.save_z(u,v,s,r)
					
					self.kalman_tracks.append(tracker)


		else:
			print('===========================================')
			print("number of real tracker is {}".format(len(self.kalman_tracks)))
			print("number of temporal tracker is {}".format(len(self.kalman_tracks_new)))

			# Keep going to prediction
			self.old_track = True
			self.kalman_run(self.kalman_tracks)
			self.old_track = False
			self.kalman_run(self.kalman_tracks_new)

			measurement_list = self.get_measurement(self.object_state)
			
			print("OLD TRACKS")
			
			iou_table = self.iou_matching(self.kalman_tracks,measurement_list)			
			self.optimal_assign(self.kalman_tracks,iou_table,measurement_list)

			print("NEW TRACKS")
			
			if self.kalman_tracks_new:
				
				iou_table_new = self.iou_matching(self.kalman_tracks_new,measurement_list)
				self.optimal_assign(self.kalman_tracks_new,iou_table_new,measurement_list)
				#print("ioutablenew\n",iou_table_new)

		



	#draw 

		for tracker in self.kalman_tracks:

			u,v,s,r = tracker.load_z()

			self.draw(u,v,s,r,'car' + str(tracker.id), tracker.color)

		cv2.imshow('window', self.cv_rgb_image)
		cv2.waitKey(3)	




	

	# def check_if_overlapped(self,box_a,box_b,iou_box):
    
	#     if (((box_a.xmin < iou_box.xmin < box_a.xmax) and (box_a.ymin < iou_box.ymin < box_a.ymax)) or \
	#     	((box_b.xmin < iou_box.xmin < box_b.xmax) and (box_b.ymin < iou_box.ymin < box_b.ymax))):
	#         return True
	#     else:
	#         return False
	def kalman_run(self,kalman_tracks):

	# Keep going to prediction
		for tracker in kalman_tracks:
			
			u,v,s,r = tracker.load_z()
			a,b,c,d = self.get_box_point(u,v,s,r)
			#self.draw(u,v,s,r,'car'+str(tracker.id),tracker.color)
			uu,vv,ss,rr = tracker.prediction(u,v,s,r)
			#self.draw(uu,vv,ss,rr,'ssssss'+str(tracker.id),tracker.color)
			# if self.old_track == True:
			# 	self.draw(uu,vv,ss,rr,'    kalman' + str(tracker.id),tracker.color)
			tracker.save_z(uu,vv,ss,rr)
			# print("$$$$$$$$$$$$$$$$$")
			# print(a,b,c,d)
			# print(aa,bb,cc,dd)

	def iou_matching(self,kalman_tracks,measurement_list):

		# #IOU matching
		iou_table = np.zeros(shape=(len(kalman_tracks), len(measurement_list)))
		#print(iou_table.shape)

		# Create IOU table
		for idx_measurement,measurement in enumerate(measurement_list):

			for idx_kalman,tracker in enumerate(kalman_tracks):
				
				u_tracker ,v_tracker ,s_tracker, r_tracker = tracker.load_z()

				u_tracker ,v_tracker ,s_tracker, r_tracker = self.get_box_point(u_tracker ,v_tracker ,s_tracker, r_tracker)

				tracker_box = box(u_tracker ,v_tracker ,s_tracker, r_tracker)



				u_measurement, v_measurement, s_measurement, r_measurement = measurement
				
				u_measurement, v_measurement, s_measurement, r_measurement = self.get_box_point(u_measurement, v_measurement, s_measurement, r_measurement)

				measuerment_box = box(u_measurement, v_measurement, s_measurement, r_measurement)

				iou_table[idx_kalman][idx_measurement] = self.iou(tracker_box,measuerment_box)

		#iou_table = -iou_table - np.min(-iou_table)
		return -iou_table

	def optimal_assign(self, kalman_tracks,iou_table,measurement_list):
		
		# Hungraian algorithm

		#print("iou table \n {}".format(iou_table))
		#print("kalman table \n {}".format())
		#print("measurement_list \n {}",measurement_list)

		row_ind, col_ind = linear_sum_assignment(iou_table)

		optimal_assignment = iou_table[row_ind,col_ind]
		#print('optimal assign {}\n'.format(optimal_assignment))
		assigned_col = []
		assigned_row = []
	
		for idx in range(len(optimal_assignment)):
			
			row, col = np.where(iou_table == optimal_assignment[idx])
			# assigned_row.append(row[0])
			# assigned_col.append(col[0])
			u_measurement = measurement_list[col[0]][0]
			v_measurement = measurement_list[col[0]][1]
			s_measurement = measurement_list[col[0]][2]
			r_measurement = measurement_list[col[0]][3]

			# Matching 
			if iou_table[row[0]][col[0]] == 0:
				pass
			else:
				assigned_row.append(row[0])
				assigned_col.append(col[0])
				kalman_tracks[row[0]].save_z(u_measurement, v_measurement, s_measurement, r_measurement)
				kalman_tracks[row[0]].hit += 1
				if kalman_tracks[row[0]].hit >= kalman_tracks[row[0]].age:
					kalman_tracks[row[0]].hit = kalman_tracks[row[0]].age
					# rospy.loginfo("car [%d]'s hit : %d",kalman_tracks[row[0]].id,kalman_tracks[row[0]].hit)
				# if kalman_tracks[row[0]].hit >= 15 and kalman_tracks[row[0]] not in self.kalman_tracks:
				#  	kalman_tracks[row[0]].hit = 15
				# 	self.kalman_tracks.append(kalman_tracks[row[0]])



		# #draw 

		# for tracker in kalman_tracks:

		# 	u,v,s,r = tracker.load_z()

		# 	self.draw(u,v,s,r,'car' + str(tracker.id), tracker.color)

	 	
		# Find unassigned objects	
		iou_row_list = np.arange(iou_table.shape[0])
		iou_col_list = np.arange(iou_table.shape[1])

		
		for row in iou_row_list:
			if row not in assigned_row:
				
				kalman_tracks[row].hit -= 1
				u,v,s,r = kalman_tracks[row].load_z()
				uu,vv,ss,rr = kalman_tracks[row].prediction(u,v,s,r)
				kalman_tracks[row].save_z(uu,vv,ss,rr)
				#rospy.loginfo("car [%d]'s hit : %d",kalman_tracks[row].id,kalman_tracks[row].hit)
				# print('hit',kalman_tracks[row].hit)
				
	
		# 	else:
		# 		pass
	
			
		# 	elif self.kalman_tracks[row].hit <= 0 :
		# 		pass

		for col in iou_col_list:

			u_measurement = measurement_list[col][0]
			v_measurement = measurement_list[col][1]
			s_measurement = measurement_list[col][2]
			r_measurement = measurement_list[col][3]

			if col not in assigned_row:
				#print('new detection',col)
				tracker = KalmanBoxTracker() # <- Tracker dltkdgka
				tracker.save_z(u_measurement,v_measurement,s_measurement,r_measurement)
				#self.draw(u_measurement,v_measurement,s_measurement,r_measurement,tracker.id,tracker.color)
				tracker.hit = 0
				rospy.logwarn("car [%d] has been added  %d col in iou table",tracker.id, col )
				self.kalman_tracks_new.append(tracker)

		# print("IOU \n {}".format(iou_table))
		for item in kalman_tracks:
			# print("assigned row {}".format(assigned_row))
			# print("assigned col {}".format(assigned_col))
			# if item in self.kalman_tracks:
			# 	rospy.loginfo("car [%d]'s hit : %d",item.id,item.hit)
			# elif item in self.kalman_tracks_new:
			# 	rospy.logerr("car [%d]'s hit : %d",item.id,item.hit)
			if item.hit < 0:
				kalman_tracks.remove(item)
				rospy.logwarn("car [%d] has been removed",item.id)

		for item in self.kalman_tracks_new:

			if item.hit >= 1:#item.age/3 : # ADDAGE
				self.kalman_tracks.append(item)
				self.kalman_tracks_new.remove(item)
				rospy.logwarn("car [%d] has been added",item.id)
				

		states_new_kalman = []
		
		for item_new_kalman in self.kalman_tracks_new:
			#print(states_new_kalman)
			states_new_kalman.append(item_new_kalman.load_z())
		
		iou_table_for_nms = self.iou_matching(self.kalman_tracks,states_new_kalman)

		
		iou_table_for_nms = np.array(iou_table_for_nms)
		#print("IOUUUUU",iou_table_for_nms,iou_table_for_nms.shape)
		for row in range(iou_table_for_nms.shape[0]):
			for col in range(iou_table_for_nms.shape[1]):
				
				#print('?',row,col,iou_table_for_nms[row][col])
				if iou_table_for_nms[row,col] != 0:
					#print("ban",row,col)
					rospy.logwarn("tracker %d has been banned ",self.kalman_tracks_new[col].id)
					#print(len(self.kalman_tracks_new))
					for idx,item in enumerate(self.kalman_tracks_new):
						#print('askdapd',self.kalman_tracks_new[idx].id)
						if self.kalman_tracks_new[col].id == item.id:
							#self.kalman_tracks_new[col].id = self.kalman_tracks[row].id
							self.kalman_tracks_new[col].hit = -5

					#self.kalman_tracks_new[col]  = copy.deepcopy(self.kalman_tracks_row[row] )
					
					#self.kalman_tracks_new.remove(self.kalman_tracks_new[col])

		


	def get_measurement(self,object_state_list):

		measurement_list = []
		for state in object_state_list:

			measurement_list.append(state[0][:4])

		return measurement_list


	    
	def check_if_overlapped(self,box_a,box_b,iou_box):

		#print((box_a.xmin < iou_box.xmin < box_a.xmax),(box_a.ymin < iou_box.ymin < box_a.ymax),(box_b.xmin <iou_box.xmin < box_b.xmax), (box_b.ymin < iou_box.ymin < box_b.ymax))
		
		# if (((box_a.xmin < iou_box.xmin < box_a.xmax) and\
		# 	(box_a.ymin < iou_box.ymin < box_a.ymax)) or\
		# 	((box_b.xmin < iou_box.xmin < box_b.xmax) and\
		# 	(box_b.ymin < iou_box.ymin < box_b.ymax))) or\
		# (((box_a.xmin < iou_box.xmax < box_a.xmax) and\
		# 	(box_a.ymin < iou_box.ymin < box_a.ymax)) or\
		# 	((box_b.xmin < iou_box.xmax < box_b.xmax) and\
		# 	(box_b.ymin < iou_box.ymin < box_b.ymax))) :

		if (box_a.xmin < iou_box.xmin < box_a.xmax) or (box_b.xmin < iou_box.xmin < box_b.xmax): 
		    return True
		else:
		    return False

	def iou(self,box_a,box_b):



		box_a_area = np.abs(box_a.xmax-box_a.xmin) * np.abs(box_a.ymax-box_a.ymin)
		box_b_area = np.abs(box_b.xmax-box_b.xmin) * np.abs(box_b.ymax-box_b.ymin)

		iou_xmin = np.max([box_a.xmin,box_b.xmin])
		iou_ymin = np.max([box_a.ymin,box_b.ymin])

		iou_xmax = np.min([box_a.xmax,box_b.xmax])
		iou_ymax = np.min([box_a.ymax,box_b.ymax])

		iou_width = np.abs(iou_xmax-iou_xmin)
		iou_height = np.abs(iou_ymax-iou_ymin)

		box_iou = box(iou_xmin,iou_ymin,iou_xmax,iou_ymax)

		# print('box a : ', box_a.xmin, box_a.ymin, box_a.xmax, box_a.ymax)
		# print('box b : ', box_b.xmin, box_b.ymin, box_b.xmax, box_b.ymax)
		# print('box iou : ', box_iou.xmin, box_iou.ymin, box_iou.xmax, box_iou.ymax)

		# self.cv_rgb_image = cv2.rectangle(self.cv_rgb_image, (int(box_iou.xmin),int(box_iou.ymin))\
		# 	, (int(box_iou.xmax), int(box_iou.ymax)), (255,255,255),-1)

		# self.cv_rgb_image = cv2.rectangle(self.cv_rgb_image, (int(box_a.xmin),int(box_a.ymin))\
		# 	, (int(box_a.xmax), int(box_b.ymax)), (255,255,0),1)


		# self.cv_rgb_image = cv2.rectangle(self.cv_rgb_image, (int(box_b.xmin),int(box_b.ymin))\
		# 	, (int(box_b.xmax), int(box_b.ymax)), (0,255,0),1)


		if self.check_if_overlapped(box_a,box_b,box_iou):
		    overlapping_region = iou_width * iou_height
		    combined_region = box_a_area + box_b_area - overlapping_region
		    IOU = overlapping_region/float(combined_region)
		 #   print('IOU : ',IOU)
		else:
		    IOU = 0
		   # print('IOU : ',IOU)

		
		return IOU




	def get_box_point(self,u,v,s,r):

		w = np.sqrt(s*r)

		h = np.sqrt(s/r)
		
		xmin = u - w/2.
		xmax = u + w/2.
		ymin = v - h/2.
		ymax = v + h/2.

		return xmin,ymin,xmax,ymax



	def draw(self, uu,vv,ss,rr,ID,color):
		blue,green,red = color[0],color[1],color[2]
		xmin, ymin, xmax, ymax = self.get_box_point(uu,vv,ss,rr)		
		self.cv_rgb_image = cv2.rectangle(self.cv_rgb_image, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (blue,green,red), 1)

		cv2.putText(
				self.cv_rgb_image,
				str(ID), 
				(int(xmin),int(ymin)-10),
				 cv2.FONT_HERSHEY_SIMPLEX, 1, (blue,green,red), 2, cv2.LINE_AA)

		
		

	def get_image(self, data):

		self.cv_rgb_image = self.bridge.imgmsg_to_cv2(data,'bgr8')

	
	# def kalman(self, states):

		
	# 	self.kf.P[4:,4:] *= 10.
	# 	self.kf.P *= 10.
	# 	u = states[0][0]
	# 	v = states[0][1]
	# 	s = states[0][2]
	# 	r = states[0][3]
	# 	ID = states[0][4]

	
	# 	x = [u,v,s,r]
	# 	self.kf.predict(x,)
	# 	self.kf.update([u,v,s,r])
		

	# 	# Kalman filter prediction

	# 	uu = self.kf.x[0]
	# 	vv = self.kf.x[1]
	# 	ss = self.kf.x[2]
	# 	rr = self.kf.x[3]
	# 	if ss < 0:
	# 		ss = 0
	# 	elif rr < 0:
	# 		rr = 0

	# 	return uu,vv,ss,rr,ID
		












if __name__=='__main__':

	Tracker()

	rospy.spin()

	