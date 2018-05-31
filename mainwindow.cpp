#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QFileDialog"
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
using namespace std;
using namespace cv;
using namespace cv::dnn;

std::vector<std::string> classes;
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame,Scalar color )
{
    rectangle(frame, Point(left, top), Point(right, bottom), color);

    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),Point(left + labelSize.width, top + baseLine), Scalar::all(255) , FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

std::vector<String> getOutputsNames(const Net& net)
{
    static std::vector<String> names;
    if (names.empty())
    {
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::vector<String> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}


void postprocess(Mat& frame, const std::vector<Mat>& outs,float confThreshold,int classidx,std::vector<Rect>& ROIs)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Network produces output blob with a shape NxC where N is a number of
        // detected objects and C is a number of classes + 4 where the first 4
        // numbers are [center_x, center_y, width, height]
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)// && classIdPoint.x==classidx)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));

                //                ROIs.push_back(Rect(left > 0 ? left : 0 , top > 0 ? top : 0  , width, height));
            }
        }
    }
    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, 0.4f, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        //        std::cout << Rect(box.x > 0 ? box.x : 0 , box.y > 0 ? box.y : 0  , box.width, box.height) << std::endl ;
        ROIs.push_back(Rect(box.x > 0 ? box.x : 0 ,
                            box.y > 0 ? box.y : 0 ,
                            box.width + (box.x > 0 ? box.x : 0) > frame.cols ? frame.cols-(box.x > 0 ? box.x : 0) : box.width,
                            box.height+ (box.y > 0 ? box.y : 0) > frame.rows ? frame.rows-(box.y > 0 ? box.y : 0) : box.height
                                                                               )
                       );
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame , Scalar(0,255,0));
    }


}

void postprocessCar(Mat& frame,Mat& input, const std::vector<Mat>& outs,float confThreshold,int classidx,cv::Rect carRect)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Network produces output blob with a shape NxC where N is a number of
        // detected objects and C is a number of classes + 4 where the first 4
        // numbers are [center_x, center_y, width, height]
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold && classIdPoint.x==classidx)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, 0.4f, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], carRect.x + box.x, carRect.y + box.y,
                 carRect.x + box.x + box.width, carRect.y + box.y + box.height, input ,Scalar(0,0,255));
    }


}



MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    String modelConfiguration = "/home/pouya/Develop/Sarbazi/yolo/darknet/cfg/yolov3.cfg";
    String modelBinary = "/home/pouya/Develop/Sarbazi/yolo/darknet/yolov3.weights";
    dnn::Net net = readNetFromDarknet(modelConfiguration, modelBinary);
    dnn::Net netPerson = readNetFromDarknet(modelConfiguration, modelBinary);

    // Open file with classes names.
    std::string file = "/home/pouya/Develop/Sarbazi/yolo/darknet/data/coco.names";
    std::ifstream ifs(file.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }


    //    VideoCapture cap;
    //    String source ="/home/pouya/1_2018-03-17_16-29-54.mp4";
    //    cap.open(source);
    //    VideoWriter writer;
    //    int codec = CV_FOURCC('M', 'J', 'P', 'G');
    //    writer.open("rest.mp4", codec, 9, Size((int)cap.get(CAP_PROP_FRAME_WIDTH),(int)cap.get(CAP_PROP_FRAME_HEIGHT)), 1);


    //    for(;;)
    //    {
    //        Mat input;
    //        cap >> input; // get a new frame from camera/video or read image
    //        if (input.empty())
    //        {
    //            waitKey();
    //            break;
    //        }

    QString fileName = QFileDialog::getOpenFileName(this,
                                                    tr("Open Address Book"), "/home/pouya/",
                                                    tr("Images (*.jpg);;All Files (*)"));
    cv::Mat input ;
    if(fileName!="") {
        input= cv::imread(fileName.toStdString().c_str()) ;
    }
    else return ;
    //            cv::Mat input = cv::imread("/home/pouya/Pictures/Screenshot from 1_2018-03-17_16-29-54.mp4 - 2.png") ;
    std::cout << input.size << std::endl ;
    cv::Mat temp ;
    input.copyTo(temp);
    Mat inputBlob = blobFromImage(input, 1 / 255.F, Size(416, 416), Scalar(), true, false); //Convert Mat to batch of images
    net.setInput(inputBlob, "data");                   //set the network input
    std::vector<Mat> outs;
    std::vector<Rect> ROIs;
    net.forward(outs, getOutputsNames(net));  //compute output


    postprocess(input, outs,0.1,2,ROIs);

    //        for (Rect ROI : ROIs) {
    //            cv::Mat ROI_Image = temp(ROI);
    //////            cvtColor(ROI_Image,ROI_Image,COLOR_BGR2GRAY);
    //////            equalizeHist( ROI_Image, ROI_Image );
    //////            cvtColor(ROI_Image,ROI_Image,COLOR_GRAY2BGR);
    //            Mat inputBlob = blobFromImage(ROI_Image, 1 / 255.F, Size(960, 960), Scalar(), true, false); //Convert Mat to batch of images
    //            net.setInput(inputBlob, "data");                   //set the network input
    //            std::vector<Mat> outs;
    //            std::vector<Rect> ROIs;
    //            net.forward(outs, getOutputsNames(net));  //compute output
    //            postprocessCar(ROI_Image,input, outs,0.1,0,ROI);
    //        }

    // Put efficiency information.
    //    std::vector<double> layersTimes;
    //    double freq = getTickFrequency() / 1000;
    //    double t = net.getPerfProfile(layersTimes) / freq;
    //    std::string label = format("Inference time: %.2f ms", t);
    //    putText(input, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

    //        if(writer.isOpened())
    //        {
    //            writer.write(input);
    //        }

    imshow("kWinName", input);
    //         if (waitKey(1) >= 0) break;
    //    }
    //    cv::imwrite("resut.jpg",input) ;
}
