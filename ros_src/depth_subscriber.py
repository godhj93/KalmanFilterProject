#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('depth_sub')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from ros_tkdnn.msg import yolo_coordinateArray

import numpy as np
class image_converter:

  def __init__(self):
  

    self.bridge = CvBridge()
    self.depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw",Image,self.depth_cb)
    self.rgb_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.rgb_cb)
    # subscribe yolo output
    self.yolo_sub = rospy.Subscriber("/yolo_output",yolo_coordinateArray,self.yolo_cb)
  def depth_cb(self,data):
    try:
      #(480,640,3)
      self.cv_depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
      
    except CvBridgeError as e:
      rospy.logerr(e)
    

  def yolo_cb(self,data):
    
  
    
    if len(data.results) > 0:
      


      # make sure the number of results
      self.yolo_output_list = []
      for idx in range(len(data.results)):
      
        self.yolo_output_list.append(
          np.array([
        data.results[idx].x_center, #0
        data.results[idx].y_center, #1
        data.results[idx].w, # 2
        data.results[idx].h, #3
        data.results[idx].xmin, #4
        data.results[idx].xmax, #5
        data.results[idx].ymin, #6
        data.results[idx].ymax, #7
        ] ,
        dtype=np.int32

        ))



    else:
      self.yolo_output_list = []
      rospy.logwarn("No results")

    self.draw()
  


  def rgb_cb(self,data):
    try:
      #(480,640,3)

      self.cv_rgb_image = self.bridge.imgmsg_to_cv2(data,'bgr8')
      self.rgb_input = True
    except CvBridgeError as e:
      self.rgb_input = False
      rospy.logerr(e)



    

  def get_depth(self,x_center,y_center):

    return self.cv_depth_image[x_center][y_center]

  def draw(self):
    
    
    self.color = (255,255,255)

    for idx in range(len(self.yolo_output_list)):
      
      
      self.cv_rgb_image = cv2.rectangle(
        self.cv_rgb_image, 
        (self.yolo_output_list[idx][4],self.yolo_output_list[idx][6]),
        (self.yolo_output_list[idx][5],self.yolo_output_list[idx][7]),

        self.color,
        3)



      
      
      self.cv_rgb_image = cv2.circle(
        self.cv_rgb_image, (self.yolo_output_list[idx][0],self.yolo_output_list[idx][1]), 3, self.color, -1)
      
      
      cv2.putText(self.cv_rgb_image,str(self.cv_depth_image[self.yolo_output_list[idx][1]][self.yolo_output_list[idx][0]]), (self.yolo_output_list[idx][0],self.yolo_output_list[idx][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2, cv2.LINE_AA)


    cv2.imshow("RGB window", self.cv_rgb_image)



    
    


    cv2.waitKey(3)



def main(args):
  ic = image_converter()
  rospy.init_node('depth_subscriber', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)