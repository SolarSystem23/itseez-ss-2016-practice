#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "opencv2/core.hpp"

using namespace std;
using namespace cv;

const char* kAbout =
    "This is an empty application that can be treated as a template for your "
    "own doing-something-cool applications.";

const char* kOptions =
    "{ v video        |        | video to process         }"
    "{ h ? help usage |        | print help message       }";


int main(int argc, const char** argv) {
  // Parse command line arguments.
  CommandLineParser parser(argc, argv, kOptions);
  parser.about(kAbout);

  // If help option is given, print help message and exit.
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }

  cv::VideoCapture video("/home/tolik/myWorkSpace/github/itseez-ss-2016-practice/test/test_data/video/logo.mp4");
  cv::namedWindow("window");
  cv::Mat frame;
  video >> frame;
  cv::imshow("window", frame);
  cv::waitKey(0);

  return 0;
}
