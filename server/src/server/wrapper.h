#ifndef SELIKA_WRAPPER_H_
#define SELIKA_WRAPPER_H_

#include "cuda/edgebundling.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <iostream>
#include <sstream>

namespace selika{

using std::vector;

//heat map properties
struct HeatMapWrapper{
  double lat_top;
  double lat_bottom;
  double lng_left;
  double lng_right;
  int grids_row;
  int grids_col;
  long int start_time;
  long int end_time;
  long int time_window;
  long int time_interval;

  HeatMapWrapper() :lat_top(0),lat_bottom(0),lng_left(0),lng_right(0),
      grids_row(0),grids_col(0),start_time(0),end_time(0),
      time_window(0),time_interval(0){}

  HeatMapWrapper(std::string str){
    std::istringstream iss(str);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(iss, pt);

    lat_top = pt.get<double>("lat_top");
    lat_bottom = pt.get<double>("lat_bottom");
    lng_left = pt.get<double>("lng_left");
    lng_right = pt.get<double>("lng_right");
    grids_row = pt.get<int>("grids_row");
    grids_col = pt.get<int>("grids_col");
    start_time = pt.get<long int>("start_time");
    end_time = pt.get<long int>("end_time");
    time_window = pt.get<long int>("time_window");
    time_interval = pt.get<long int>("time_interval");
  }
};

struct RadarWrapper{
  long int start_time;
  long int end_time;
  long int time_window;
  long int time_interval;
  double r;
  double lat;
  double lng;
  int arg,id;
  
  RadarWrapper(std::string str){
    std::istringstream iss(str);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(iss, pt);

    start_time = pt.get<long int>("start_time");
    end_time = pt.get<long int>("end_time");
    time_window = pt.get<long int>("time_window");
    time_interval = pt.get<long int>("time_interval");
    r = pt.get<double>("r");
    lng = pt.get<double>("lng");
    lat = pt.get<double>("lat");
    arg = pt.get<int> ("arg");
    id = pt.get<int> ("id");
  }
};

struct EdgeBundlingWrapper{
  long int start_time;
  long int end_time;
  long int time_window;
  long int time_interval;
  double lat_top;
  double lat_bottom;
  double lng_left;
  double lng_right;
  double K,S,I_rate,threshold;
  int P,P_rate,C,I;

  EdgeBundlingWrapper(std::string str){
    std::istringstream iss(str);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(iss,pt);

    lat_top = pt.get<double>("lat_top");
    lat_bottom = pt.get<double>("lat_bottom");
    lng_left = pt.get<double>("lng_left");
    lng_right = pt.get<double>("lng_right");
    start_time = pt.get<long int>("start_time");
    end_time = pt.get<long int>("end_time");
    time_window = pt.get<long int>("time_window");
    time_interval = pt.get<long int>("time_interval");
    K = pt.get<double>("K");
    S = pt.get<double>("S");
    P = pt.get<int>("P");
    P_rate = pt.get<int>("P_rate");
    C = pt.get<int>("C");
    I = pt.get<int>("I");
    I_rate = pt.get<double>("I_rate");
    threshold = pt.get<double>("threshold");
  }
};

struct GeoObjWrapper{
  int id;
  double radius, left_lng, right_lng, top_lat, bottom_lat, cnt_lng, cnt_lat;
  long start_time, end_time;
  GeoObjWrapper(std::string str){
    std::istringstream iss(str);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(iss,pt);
    id = pt.get<int>("id");
    radius = pt.get<double>("radius",-1.f);
    left_lng = pt.get<double>("left_lng",-1.f);
    right_lng = pt.get<double>("right_lng",-1.f);
    top_lat = pt.get<double>("top_lat",-1.f);
    bottom_lat = pt.get<double>("bottom_lat",-1.f);
    cnt_lng = pt.get<double>("cnt_lng",-1.f);
    cnt_lat = pt.get<double>("cnt_lat",-1.f);
    start_time = pt.get<long>("start_time",0);
    end_time = pt.get<long>("end_time",2e10);
  }
};

struct ChordMatrixWrapper{
  int id;
  long start_time, end_time;
  ChordMatrixWrapper(std::string str){
    std::istringstream iss(str);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(iss,pt);

    id = pt.get<int>("id");
    start_time = pt.get<long>("start_time",0);
    end_time = pt.get<long>("end_time",2e10);
  }
};

struct ChordWrapper{
  int id;
  ChordWrapper(std::string str){
    std::istringstream iss(str);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(iss, pt);

    id = pt.get<int>("id");
  }
};

struct POIWrapper{
  vector<std::string> *s;
  POIWrapper(std::string str) {
    std::istringstream iss(str);
    
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(iss,pt);
    
  }
};

struct MessageWrapper{
  int op;
  MessageWrapper(std::string str){
    std::istringstream iss(str);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(iss, pt);
    op = pt.get<int>("op");
  }
};

//heat map render data
struct HeatMapDataWrapper{
  const int* memPtr;
  int length;
  std::string data_str;
  //HeatMapDataWrapper() :memPtr(), length(0), data_str(){}
  HeatMapDataWrapper(const int* array, int length){
    if (array == nullptr) return;
    this->length = length;
    std::stringstream ss("");
    memPtr = array;
    if (length > 0) ss << "[" << "0,";
    int i;
    for (i = 0; i < length-1; i++){
	ss << std::to_string(memPtr[i]) << ",";
    }
    if (length>0) ss << std::to_string(memPtr[i]) << "]";
    data_str = ss.str();
  }
};

struct RadarDataWrapper{
  const int* memPtr;
  int length;
  std::string data_str;
  //HeatMapDataWrapper() :memPtr(), length(0), data_str(){}
  RadarDataWrapper(const int* array, int length,int id){
    if (array == nullptr) return;
    this->length = length;
    std::stringstream ss("");
    memPtr = array;
    if (length > 0) {
      ss << "[" << "1,";
      ss << std::to_string(id) << ",";
    }
    int i;
    for (i = 0; i < length-1; i++){
        ss << std::to_string(memPtr[i]) << ",";
    }
    if (length>0) ss << std::to_string(memPtr[i]) << "]";
    data_str = ss.str();
  }
};

struct EdgeDataWrapper{
  std::string data_str;

  EdgeDataWrapper(const vector<vector<edgebundling::Node*>> &data){
    std::stringstream ss("");
    ss << "[" <<"2";
    std::cout<<"data size: "<<data.size()<<std::endl;
    for (int i=0;i<data.size();i++){
      ss<< ",";
      for (int j=0;j<data[i].size();j++){
        ss<< std::to_string(data[i][j]->y)<<",";
        ss<< std::to_string(data[i][j]->x)<<",";
      }
      ss<< "-1";
    }
    ss<<"]";
    data_str=ss.str();
  }
};

struct ChordDataWrapper{
  std::string data_str;

  ChordDataWrapper(int id, const int *array, int len){
    std::stringstream ss("");
    if (len <= 0) {
      ss<<"[]";
      data_str = ss.str();
      return;
    }
    ss << "[" <<"5,"<<std::to_string(id);
    for (int i=0; i<len; i++){
      ss<<","<<std::to_string(array[i]);
    }
    ss<<"]";
    data_str=ss.str();
  }
};
}

#endif  //SELIKA_WRAPPER_H_
