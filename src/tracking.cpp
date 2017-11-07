#include "tracking.hpp"

#include <iostream>
#include <opencv/cv.hpp>

using std::string;
using std::shared_ptr;
using namespace std;
using namespace cv;

shared_ptr<Tracker> Tracker::CreateTracker(const string &name) {
  if (name == "median_flow") {
    return std::make_shared<MedianFlowTracker>();
  }
  else
    throw "name != median_flow";
}

bool MedianFlowTracker::Init(const cv::Mat &frame, const cv::Rect &roi) {
    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, CV_BGR2GRAY);
    frame_= frame;
    position_ = roi;
    return !frame_.empty();
}

cv::Rect MedianFlowTracker::Track(const cv::Mat &frame) {
    cvtColor(frame_, frame_, CV_BGR2GRAY);
    int maxCorners = 100;
    double qualityLevel = 0.3;
    int minDistance = 7;
    std::vector<Point2f> prevPts;
    std::vector<Point2f> nextPts;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, CV_BGR2GRAY);
    cv::Mat roiPrev = frame_(position_);
    goodFeaturesToTrack(roiPrev, prevPts, maxCorners,
                        qualityLevel, minDistance);

    for(const auto& point : prevPts){
        cv::circle(roiPrev, point, 5, cv::Scalar(0, 255, 0));
    }

    calcOpticalFlowPyrLK(frame_, gray_frame, prevPts, nextPts, status, err);

    for(const auto& point : nextPts){
        cv::circle(frame, point, 2, cv::Scalar(0, 0, 255));
    }

    for(int i = 0 ; i < status.size(); i++){
        if(!status[i]){
            nextPts.erase(nextPts.begin() + i);
            prevPts.erase(prevPts.begin() + i);
        }
    }
    std::vector<Point2f> backwards;
    status.clear();
    err.clear();

    calcOpticalFlowPyrLK(gray_frame, frame_, nextPts, backwards, status, err);
    //static_assert(nextPts.size() == backwards.size());
    int shiftSize = nextPts.size();
    std::vector<double>shift(shiftSize);

    for(int i = 0; i < shiftSize; i++){
        shift[i] = cv::norm(nextPts[i] - backwards[i]);
    }
    std::vector<double> copyShift(shiftSize);
    std::copy(shift.begin(), shift.end(), copyShift.begin());
    double medianShift;
    std::nth_element(shift.begin(), shift.begin() + shiftSize / 2, shift.end());
    for(int i = 0 ; i < shiftSize; i++){
        if(copyShift[i] > medianShift){
            prevPts.erase(prevPts.begin() + i);
            nextPts.erase(nextPts.begin() + i);
        }
    }

    //static_assert(nextPts.size() == prevPts.size());
    int sizeOfVectors = nextPts.size();
    std::vector<double> distancePrev;
    std::vector<double> distanceNext;

    for(int i = 0 ; i < sizeOfVectors - 1; i++ ){
        for(int j = i + 1; j < sizeOfVectors; j++){
            distancePrev.push_back(cv::norm(prevPts[i] - prevPts[j]));
            distanceNext.push_back(cv::norm(nextPts[i] - nextPts[j]));
        }
    }

    std::vector<double> scales(distanceNext.size());
    int sizeOfScales = scales.size();
    for(int i = 0 ; i < sizeOfScales; i++ ){
        scales[i] = distancePrev[i] / distanceNext[i];
    }
    std::nth_element(scales.begin(), scales.begin() + sizeOfScales / 2, scales.end());

    double median = scales[sizeOfScales / 2];
    auto width  = saturate_cast<int>(median * position_.width);
    auto height = saturate_cast<int>(median * position_.height);

    std::vector<double> distanceX(sizeOfVectors);
    std::vector<double> distanceY(sizeOfVectors);
    for(int i = 0 ; i < sizeOfVectors; i++){
        distanceX[i] = prevPts[i].x - nextPts[i].x;
        distanceY[i] = prevPts[i].y - nextPts[i].y;
    }

    double shiftX, shiftY;
    std::nth_element(distanceX.begin(), distanceX.begin() + sizeOfVectors / 2, distanceX.end());
    std::nth_element(distanceY.begin(), distanceY.begin() + sizeOfVectors / 2, distanceY.end());

    shiftX = distanceX[sizeOfVectors / 2];
    shiftY = distanceY[sizeOfVectors / 2];

    frame_    = frame;
    position_ = cv::Rect(position_.x + shiftX, position_.y + shiftY, width, height);;
    return position_;
}
