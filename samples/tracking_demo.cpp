#include <iostream>
#include <string>
#include <opencv2/videoio.hpp>
#include <opencv/cv.hpp>
#include <tracking.hpp>

#include "opencv2/core.hpp"

using namespace std;
using namespace cv;

const char* kOptions =
                "{ v video        | <none> | video to process                         }"
                "{ c camera       | <none> | camera to get video from                 }"
                "{ h ? help usage |        | print help message                       }";


struct MouseCallbackState {
    bool is_selection_started;
    bool is_selection_finished;
    Point point_first;
    Point point_second;
} mouseCallbackState;
void onMouse(int event, int x, int y, int flag, void* param)
{
    if(event == cv::EVENT_LBUTTONDOWN)
    {
        mouseCallbackState.is_selection_started = true;
        mouseCallbackState.is_selection_finished = false;
        mouseCallbackState.point_first = cv::Point(x, y);
    }
    if(event == cv::EVENT_LBUTTONUP)
    {
        mouseCallbackState.is_selection_started = false;
        mouseCallbackState.is_selection_finished = true;
        mouseCallbackState.point_second = cv::Point(x, y);
    }
    if(event == cv::EVENT_MOUSEMOVE && !mouseCallbackState.is_selection_finished)
    {
        mouseCallbackState.point_second = cv::Point(x, y);
    }
}

int main(int argc, const char** argv) {
  // Parse command line arguments.
  CommandLineParser parser(argc, argv, kOptions);

  // If help option is given, print help message and exit.
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }

  std::string filePath;
 if(parser.has("v")) {
   Mat frame;
   std::string filePath;
   filePath = parser.get<std::string>("v");
   cv::VideoCapture video(0);

   //cv::CascadeClassifier detector("/home/luba/github/itseez-ss-2016-practice/logo_cascade/haarcascade_frontalface_alt.xml");
   //std::vector<cv::Rect> objects;
   MedianFlowTracker tracker;
   video >> frame;
   cvtColor(frame, frame, CV_BGR2GRAY);
   tracker.Init(frame, cv::Rect(300, 300, 100, 100));
   while (true) {
       video >> frame;
       cvtColor(frame, frame, CV_BGR2GRAY);
       tracker.Track(frame);
       cvtColor(frame, frame, CV_BGR2GRAY);
       char c  = cv::waitKey(33);
       if(c == 27)
           break;

   }

     video.release();
 }

    else {
     cerr << "Error load model";
     return -1;
 }


  return 0;
}
