#ifndef SELIKA_RADAR_RADAR_H_
#define SELIKA_RADAR_RADAR_H_
#include "datastruct.h"
#include "comdb.h"
#include <thrust/device_vector.h>
#include <thrust/complex.h>

namespace selika{
namespace radar{
struct radar_divide{
  double *div_vec;
  double lat,lng,r;
  int arg;
  const double PI = 3.14159265358;
  radar_divide(const double center_lat, const double center_lng,
      const double r, double *div_vec, const int a)
      :lat(center_lat),lng(center_lng),arg(a), r(r), div_vec(div_vec){}

  template <typename T>
  __device__
  void operator()(const T& i){
    div_vec[2*i]=r*sin(2*PI*i/arg-PI/arg);
    div_vec[2*i+1]=r*cos(2*PI*i/arg-PI/arg);
  }
};

struct QueryCell{
  long int start_time, end_time, time_window, time_interval;
  double r, lat, lng;
  int arg;
  double *div_vec;
  const double *lng_data, *lat_data;
  const long int *time_data;
  const int *id;
  int *data_id, *cell_vec;

  QueryCell(long int start_time, long int end_time, long int time_window
      , long int time_interval, double r, double lat, double lng, int arg
      , double *div_vec, const double *lat_data, const double *lng_data
      , const long int *time, const int *id, int *data_id, int *cell_vec)
      : start_time(start_time), end_time(end_time), time_window(time_window)
      , time_interval(time_interval), r(r), lat(lat), lng(lng), arg(arg)
      , div_vec(div_vec), lng_data(lng_data), lat_data(lat_data)
      , time_data(time), id(id), data_id(data_id), cell_vec(cell_vec){}
  __host__ __device__
  bool inRange(double x1,double y1, double x2, double y2){
    if ((x1-lng)*(x1-lng)+(y1-lat)*(y1-lat)<=r*r) return true;
    if ((y2-y1)*(y2-y1)+(x2-x1)*(x2-x1)==0) return false;
    double dis2=((y2-y1)*lng-(x2-x1)*lat+x2*y1-y2*x1)*
               ((y2-y1)*lng-(x2-x1)*lat+x2*y1-y2*x1)/
               ((y2-y1)*(y2-y1)+(x2-x1)*(x2-x1));
    if (dis2>r*r) return false;
    double line_len2 = (y2-y1)*(y2-y1)+(x2-x1)*(x2-x1);
    double part1_len2 = (x1-lng)*(x1-lng)+(y1-lat)*(y1-lat)-dis2;
    double part2_len2 = (x2-lng)*(x2-lng)+(y2-lat)*(y2-lat)-dis2;
    if (part1_len2>line_len2||part2_len2>line_len2) return false;
    return true;
  }

  template <typename T>
  __host__ __device__
  void operator()(const T& i){
    long int time=time_data[i];
    int id1=id[i], id2=id[i+1];
    double start_lng=lng_data[i], end_lng=lng_data[i+1];
    double start_lat=lat_data[i], end_lat=lat_data[i+1];

    if (time>=start_time&&time<end_time&&id1==id2
        &&inRange(start_lng,start_lat,end_lng,end_lat)){

      double x0=end_lng-start_lng;
      double y0=end_lat-start_lat;

      int id_index=-1, cell_index=-1;
      for (int j=0; j<arg; j++){
        double x1=div_vec[2*j], x2=div_vec[2*((j+1)%arg)],
               y1=div_vec[2*j+1], y2=div_vec[2*((j+1)%arg)+1];
        /*if (x1>=0&&x2>=0&&x0>=0&&(x0*y1>=x1*y0)&&(x2*y0>x0*y2)){
          id_index = id1;
          cell_index = j;
          continue;
        }
        if (x1<0&&x2<0&&x0<0&&x0*y1>=x1*y0&&x2*y0>x0*y2){
          id_index = id1;
          cell_index = j;
          continue;
        }
        if (x1*x2<0){
          if (0<=x0&&x0*y1>=x1*y0){
            id_index = id1;
            cell_index = j;
            continue;
          }
          if (0>x0&&x2*y0>x0*y2){
            id_index = id1;
            cell_index = j;
            continue;
          }
        }*/
        if (x1*y0-x0*y1<=0&&x0*y2-x2*y0<0){
          id_index = id1;
          cell_index=j;
        }
      }
//      printf("id_index: %d, cell_index: %d\n",id_index, cell_index);
      int z=ceilf((time-time_window-start_time)/time_interval);
      int duplicate_time=ceilf(time_window/time_interval);
      for (int j=z;j<=(time-start_time)/time_interval;j++){
        if (j<0) continue;
        if (j-z>=duplicate_time) break;
        if (j>=1+ceilf((end_time-start_time-time_window)/time_interval)) break;
        data_id[i*duplicate_time+j-z]=id_index;
        cell_vec[i*duplicate_time+j-z]=cell_index+j*arg;
      }
    }
  }
};

struct EqualPlus{
  int *keys;
  int *values;
  int *counts;

  EqualPlus(){}
  EqualPlus(int *k, int *v, int *c):keys(k),values(v),counts(c){}

  template <typename T>
  __host__ __device__
  void operator() (const T& i){
    if (keys[i]<0) return;
    if (i==0) counts[keys[i]]=1;
    else{
      if (keys[i]>0&&keys[i]!=keys[i-1]){
        atomicAdd(counts+keys[i],1);
      }else{
        if (values[i]>0&&values[i]!=values[i-1])
          atomicAdd(counts+keys[i],1);
      }
    }
  }
};

class Radar{
 private:
  long int start_time, end_time, time_window, time_interval;
  double r,lat,lng;
  int arg;
  thrust::device_vector<double> div_vec;

 public:
  Radar();
  Radar(long int start_time, long int end_time, long int time_window
      , long int time_interval, double r, double lat, double lng, int arg);
//  Radar(long int start_time, long int end_time, BoundingBox bbox, int arg);
  int init();
  int update();
  int* query_radar(const comdb &db);

};
}  //radar
}  //selika

#endif  //SELIKA_RADAR_RADAR_H_
