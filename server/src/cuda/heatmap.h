/*
*Created by Zhang Lei
*divide to 10,000 cells to calculate heatmap
*/
#ifndef SELIKA_HEATMAP_HEATMAP_H_
#define SELIKA_HEATMAP_HEATMAP_H_
#include <thrust/device_vector.h>

#include"comdb.h"

namespace selika{
namespace heatmap{

struct QueryCell{
  double lat_top,lat_bottom,lng_left,lng_right;
  int grids_row,grids_col;
  long int start_time,end_time,time_window,time_interval;
  const int *id;
  int *data_id,*cell_vec;
  const double *lat,*lng;
  const long int *time;
  QueryCell(){}
  QueryCell(double lat_top, double lat_bottom, double lng_left
      , double lng_right, int grids_row, int grids_col, long int start_time
      , long int end_time, long int time_window, long int time_interval
      , const double *lng, const double *lat, const long int *time
      , const int *id,int *data_id, int*cell_vec )
          :lat_top(lat_top),lat_bottom(lat_bottom),lng_left(lng_left)
          ,lng_right(lng_right),grids_row(grids_row),grids_col(grids_col)
          ,start_time(start_time),end_time(end_time)
          ,time_window(time_window),time_interval(time_interval)
          ,lng(lng),lat(lat),time(time),id(id),data_id(data_id)
          ,cell_vec(cell_vec){}
  template <typename T>
  __host__ __device__
  void operator()(const T& i){
    //judge if one point is in a particular cell
    if (time[i]>=start_time && time[i]<end_time&&lat[i]>lat_bottom
           &&lat[i]<=lat_top&&lng[i]>=lng_left&&lng[i]<lng_right){
      double cell_lng=(lng_right-lng_left);
      double cell_lat=(lat_top-lat_bottom);
      int x=floor(grids_row*(lng[i]-lng_left)/cell_lng);
      int y=floor(grids_col*(lat_top-lat[i])/cell_lat);
      int z=ceilf((time[i]-time_window-start_time)/time_interval);
      int duplicate_time=ceilf(time_window/time_interval);
      for (int j=z;j<=(time[i]-start_time)/time_interval;j++){
        if (j<0) continue;
        if (j-z>=duplicate_time) break;
        if (j>=1+ceilf((end_time-start_time-time_window)/time_interval)) break;
        data_id[i*duplicate_time+j-z]=id[i];
        cell_vec[i*duplicate_time+j-z]=x+y*grids_row+j*grids_row*grids_col;
      }
    }
  }
};

//structure to calculate value of sum without duplication
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

//heat map class after time slice added
class HeatMap{
 private:
  double lat_top,lat_bottom,lng_left,lng_right;
  int grids_row,grids_col;
  long int start_time,end_time,time_window,time_interval;

 public:
  HeatMap(double lat_top, double lat_bottom, double lng_left, double lng_right
      , int grids_row, int grids_col, long int start_time, long int end_time
      , long int time_window, long int time_interval);
  int* query_heatmap(const comdb &db);
};
}  //heatmap
}  //selika

#endif  //SELIKA_HEATMAP_HEATMAP_H_
