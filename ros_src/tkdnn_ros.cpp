#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
//#include <unistd.h>
#include <mutex>

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"


bool gRun;
bool SAVE_RESULT = false;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

// ROS Image converter for YOLO input
#include "ros/ros.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros_tkdnn/yolo_coordinate.h>
#include <ros_tkdnn/yolo_coordinateArray.h>

#define CAMERA 1
#define VIDEO 0
cv_bridge::CvImagePtr cv_ptr;
class ImageConverter
{
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;

    image_transport::Subscriber image_sub_;

public:
    ImageConverter()
    : it_(nh_)
    {
        image_sub_ = it_.subscribe("/camera/color/image_raw",10, &ImageConverter::imageCb,this);

    }
    void imageCb(const sensor_msgs::ImageConstPtr& msg){
        
        try
        {
            
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
};


int main(int argc, char *argv[]) {


    std::cout<<"detection\n";
    signal(SIGINT, sig_handler);

    ROS_INFO("Initialize ROS");
    ros::init(argc, argv, "image_converter");


    ros::NodeHandle nh;
    
    
    ImageConverter ic; // For ROS image input


    //Publish video frame for kalman filter project
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("video/image", 1);

    ros::Publisher yolo_output = nh.advertise<ros_tkdnn::yolo_coordinateArray>("yolo_output",10);
    ROS_INFO("add publisher : yolo_output");
    uint rate = 100;
    ros::Rate loop_rate(rate);    
    

    // Initialize deep network of darknet(tkdnn)
    std::string videoPath;
    std::string weightsModel;
    int numClasses;
    int input_mode = -1; // 1: camera, 0 : video

    nh.getParam("yolo_model/input_mode/value", input_mode);
    if(input_mode == CAMERA){
        ROS_INFO("input mode : camera");
    }
    else if(input_mode == VIDEO){
        ROS_INFO("input mode : video");
    }
    else{
        ROS_ERROR("input mode must be 1(CAMERA) or 0(VIDEO)");
    }
    // Path to video file
    nh.getParam("yolo_model/video_file/name", videoPath);

    // Path to weights file
    nh.getParam("yolo_model/weights_file/name", weightsModel);
    ROS_INFO("weightsModel: %s", weightsModel.c_str());

    // get class size
    nh.getParam("yolo_model/detection_classes/value", numClasses);
    
    ROS_INFO("class : %d", numClasses);
    // Threshold of object detection
    float thresh;
    nh.getParam("yolo_model/threshold/value", thresh);
    
    ROS_INFO("threshold : %f", thresh);




    std::string net = weightsModel;
    std::string input = videoPath;
    char ntype = 'y';
    int n_classes = numClasses;
    int n_batch = 1;
    bool show = false;
    float conf_thresh=thresh;    
    ROS_INFO("ntype : %c, batch size : %d",ntype, n_batch);

    if(n_batch < 1 || n_batch > 64)
        ROS_ERROR("Batch dim not supported");

    if(!show)
        SAVE_RESULT = true; // make it ros parameter 
    
    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;
    tk::dnn::MobilenetDetection mbnet;  

    tk::dnn::DetectionNN *detNN;  

    ROS_INFO("Algorithm : YOLO");
    detNN = &yolo;

    ROS_INFO("initialize YOLO");
    detNN->init(net, n_classes, n_batch, conf_thresh); 

    
    gRun = true;

    cv::VideoCapture cap(input);
    if(!cap.isOpened())
        gRun = false; 
    else
        ROS_INFO("camera started\n");

    cv::VideoWriter resultVideo;
    if(SAVE_RESULT) {
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        resultVideo.open("result.mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(w, h));
    }

    cv::Mat frame;
    if(show)
        cv::namedWindow("detection", cv::WINDOW_NORMAL);
    

    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;

    std::vector<tk::dnn::GodHJBox> box_ary;
    ros_tkdnn::yolo_coordinate output;

    ROS_INFO("Start Detection");
    while(gRun && ros::ok()) {
        
        double begin = ros::Time::now().toSec(); // for FPS

        ros_tkdnn::yolo_coordinateArray output_array;
       
        output.header.stamp = ros::Time::now();
        
        for(auto&&b : box_ary){
            output.x_center = b.x_center;
            output.y_center = b.y_center;
            output.w = b.w;
            output.h = b.h;
            output.label = b.label;
            output.confidence = b.confidence;
           // output.size = b.size;
            output.xmin = b.xmin;
            output.xmax = b.xmax;
            output.ymin = b.ymin;
            output.ymax = b.ymax;
            output.size = b.size;
            output_array.results.push_back(output);
        }
        
       
        yolo_output.publish(output_array);
        
        ros::spinOnce(); 
        
        loop_rate.sleep();
       
        batch_dnn_input.clear();
     
        batch_frame.clear();
       
        
        
        

        for(int bi=0; bi< n_batch; ++bi){
            
            if(input_mode == CAMERA){frame=(cv_ptr->image);}
            else if(input_mode == VIDEO){cap >> frame;}
            else{ROS_ERROR("input is neither camera or video");}
            
            if(!frame.data) 
                break;
            
            batch_frame.push_back(frame);

            // this will be resized to the net format
            batch_dnn_input.push_back(frame.clone());
        } 
        if(!frame.data)
            break;
    
        //inference
        detNN->update(batch_dnn_input, n_batch);
        detNN->draw2(batch_frame, box_ary);

        //Publish image
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", batch_frame[0]).toImageMsg();
        pub.publish(msg);

        if(show){
            for(int bi=0; bi< n_batch; ++bi){
                cv::imshow("detection", batch_frame[bi]);
                cv::waitKey(1);
            }
        }
        if(n_batch == 1 && SAVE_RESULT)
            resultVideo << frame;

        double fin = ros::Time::now().toSec(); // for FPS // for FPS
        
        ROS_INFO("time : %f, FPS : %d", (fin-begin), int(1/(fin-begin))); // for FPS
    }

    std::cout<<"detection end\n";   
    double mean = 0; 
    
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
   std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    for(int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean/n_batch<<" ms\t"<<1000/(mean/n_batch)<<" FPS\n"<<COL_END;   
    

    return 0;
}

