#ifndef SELIKA_DATASTRUCT_H_
#define SELIKA_DATASTRUCT_H_

namespace selika{
struct BoundingBox{
  double lat_top;
  double lng_left;
  double lat_bottom;
  double lng_right;
  BoundingBox(double lt, double ll, double lb, double lr)
      :lat_top(lt),lng_left(ll),lat_bottom(lb),lng_right(lr){}
};
}
#endif  //SELIKA_DATASTRUCT_H_
