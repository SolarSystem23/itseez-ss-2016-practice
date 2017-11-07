#include "tracking.hpp"

#include <iostream>
#include <opencv/cv.hpp>

using std::string;
using std::shared_ptr;
using namespace cv;

shared_ptr<Tracker> Tracker::CreateTracker(const string &name) {
  if (name == "median_flow") {
    return std::make_shared<MedianFlowTracker>();
  }
  else
    throw "name != median_flow";
}


bool MedianFlowTracker::Init(const cv::Mat &frame, const cv::Rect &roi) {
    frame_= frame;
    position_ = roi;
    return !frame_.empty();
}

cv::Rect MedianFlowTracker::Track(const cv::Mat &frame) {
    int maxCorners = 100;
    double qualityLevel = 0.3;
    int minDistance = 7;
//    int blockSize = 7;
    std::vector<Point> prevPts;
    std::vector<Point> nextPts;
    std::vector<uchar > status;
    std::vector<float> err;
    cv::Mat roiPrev = frame_(position_);
    goodFeaturesToTrack(roiPrev, prevPts, maxCorners,
                        qualityLevel, minDistance);

    for(const auto& point : prevPts){
        cv::circle(roiPrev, point, 5, cv::Scalar(0, 255, 0));
    }

    calcOpticalFlowPyrLK(roiPrev, frame, prevPts, nextPts, status, err);

    for(const auto& point : nextPts){
        cv::circle(frame, point, 5, cv::Scalar(0, 0, 255));
    }

    for(int i = 0 ; i < status.size(); i++){
        if(!status[i]){
            nextPts.erase(nextPts.begin() + i);
            prevPts.erase(prevPts.begin() + i);
        }
    }
    static_assert(nextPts.size() == prevPts.size());
    int sizeOfVectors = nextPts.size();
    std::vector<double> distancePrev;
    std::vector<double> distanceNext;

    for(int i = 0 ; i < sizeOfVectors - 1; i++ ){
        for(int j = i + 1; j < sizeOfVectors; j++){
            //distancePrev.push_back();
        }
    }

    std::vector<Point> distance;
    for(int i = 0 ; i < prevPts.size(); i++){
        distance[i].x = nextPts[i].x - prevPts[i].x;
        distance[i].y = nextPts[i].y - prevPts[i].y;
    }
    size_t n = distance.size() / 2;


    Point median;

    std::nth_element(distance.begin(), distance.begin() + n, distance.end(), [](const Point& p1,
                                                                                const Point& p2){
        return p1.x < p2.x;
    });

    median.x = distance[n].x;

    std::nth_element(distance.begin(), distance.begin() + n, distance.end(), [](const Point& p1,
                                                                                const Point& p2){
        return p1.y < p2.y;
    });

    median.y = distance[n].y;


    return cv::Rect(median.x - position_.width / 2, median.y - position_.height / 2,
                    position_.width, position_.height);
}
