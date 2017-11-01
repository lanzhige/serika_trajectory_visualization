#ifndef SELIKA_EDGEBUNDLING_H__
#define SELIKA_EDGEBUNDLING_H__

#include "comdb.h"
#include "lexicographical_sort.h"

#include <thrust/copy.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>


#include <vector>
#include <iostream>

namespace selika{
namespace edgebundling{
const double eps = 1e-8;
using std::vector;

struct Node {
  double x;
  double y;
  Node(){}
  Node(double x, double y):x(x),y(y){}
  Node(const Node &obj){
    if (this != &obj){
      x=obj.x;
      y=obj.y;
    }
    std::cerr<<"use constructor 1"<<std::endl;
  }
  Node &operator=(const Node &obj){
    if (this != &obj){
      this->x=obj.x;
      this->y=obj.y;
    }
    std::cerr<<"use constructor 2"<<std::endl;
    return *this;
  }
};

struct Edge {
  Node *source;
  Node *target;
  Edge(Node *s, Node *t) {
    if (source!=s) {
      source = s;
    } 
    if (target!=t) {
      target = t;
    }
  }
  Edge(const Edge &obj){
    if (this != &obj){
      source=obj.source;
      target=obj.target;
    }
  }
};

void dataFetch(const comdb &db/*, vector<Node> *data_nodes*/
    , vector<Edge> *data_edges, double lat_top, double lat_bottom
    , double lng_left, double lng_right, long int start_time
    , long int end_time, long int time_window, long int time_interval);

class EdgeBundling{
 public:
  EdgeBundling();
  EdgeBundling(
      /*vector<Node> &data_nodes,*/ vector<Edge> &data_edges
      , double K = 0.1, double S = 0.1, int P = 1,int P_rate = 2, int C = 6
      , int I = 70, double I_rate = 0.6666667
      , double compatibility_threshold =0.6
      , bool invers_quadratic_mode = false);
  vector<vector<Node*>> outputPoints(){
    for (int i=0;i<subdivision_points_for_edge.size();i++)
      for (int j=0;j<subdivision_points_for_edge[i].size();j++){
        subdivision_points_for_edge[i][j]->x/=10000.0;
        subdivision_points_for_edge[i][j]->y/=10000.0;
      }
    return subdivision_points_for_edge;
  };

  void forceBundle();

 private:
  void initializeEdgeSubdivisions(int P_initial);
  void initializeCompatibilityLists();
  void computeCompatibilityLists();
  void updateEdgeDivisions(int P);

  //------math functions
  double vectorDotProduct(const Node *p, const Node *q);
  Node *edgeAsVector(const Edge &P);
  double edgeLength(const Edge &e);
  double customEdgeLength(const Edge &e);
  Node *edgeMidpoint(const Edge &e);
  double computeDividedEdgeLength(int e_idx);
  double euclideanDistance(const Node *p,const Node *q);
  Node *projectPointOnLine(const Node *p, const Edge &Q);

  //------force calculation methods
  Node *applySpringForce(int e_idx, int i, double kP);
  Node *applyElectrostaticForce(int e_idx, int i);
  vector<Node*> applyResultForcesOnSubdivisionPoints(
      int e_idx, int P, double S);

  //------compatible calculation
  double angleCompatibility(const Edge &P, const Edge &Q);
  double scaleCompatibility(const Edge &P, const Edge &Q);
  double positionCompatibility(const Edge &P, const Edge &Q);
  double edgeVisibility(const Edge &P, const Edge &Q);
  double visibilityCompatibility(const Edge &P, const Edge &Q);
  double edgeProximity(const Edge &P, const Edge &Q);
  double proximityCompatibility(const Edge &P, const Edge &Q);
  double compatibilityScore(const Edge &P, const Edge &Q);
  bool areCompatible(const Edge &P, const Edge &Q);

  //const vector<Node> data_nodes;
  vector<Edge> data_edges;
  double K, S, I_rate, compatibility_threshold;
  int P, P_rate, C, I;
  bool invers_quadratic_mode;
  vector<vector<int>> compatibility_list_for_edge;
  vector<vector<Node*>> subdivision_points_for_edge;
};
}  //edgebundling
}  //selika
#endif  //SELIKA_EDGEBUNDLING_H__
