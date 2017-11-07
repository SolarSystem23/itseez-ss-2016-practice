#include "tracking.hpp"

#include <iostream>

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
