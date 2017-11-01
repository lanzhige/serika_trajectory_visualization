#include "edgebundling.h"

#include <thrust/system/tbb/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <math.h>
#include <algorithm>
#include <iostream>

#define MAX_DELTA_TIME 900

namespace selika{
namespace edgebundling{
void dataFetch(const comdb &db/*, vector<Node> *data_nodes*/
    , vector<Edge> *data_edges, double lat_top, double lat_bottom
    , double lng_left, double lng_right, long int start_time
    , long int end_time, long int time_window, long int time_interval){
  int col_len = db.col_id_vec.size();
  thrust::device_vector<int> col_id_vec(col_len);
  thrust::device_vector<long int> col_time_vec(col_len);
  thrust::device_vector<double> col_lat_vec(col_len);
  thrust::device_vector<double> col_lng_vec(col_len);
  thrust::copy_n(db.col_id_vec.begin(),col_len,col_id_vec.begin());
  thrust::copy_n(db.col_time_vec.begin(),col_len,col_time_vec.begin());
  thrust::copy_n(db.col_lat_vec.begin(), col_len, col_lat_vec.begin());
  thrust::copy_n(db.col_lon_vec.begin(), col_len, col_lng_vec.begin());

  thrust::device_vector<int> permutation(col_len);
  thrust::sequence(permutation.begin(),permutation.end());

  update_permutation(col_time_vec, permutation);
  update_permutation(col_id_vec, permutation);

  apply_permutation(col_id_vec,permutation);
  apply_permutation(col_time_vec,permutation);
  apply_permutation(col_lat_vec,permutation);
  apply_permutation(col_lng_vec,permutation);
//should be moved to data initialize and filter
  int *id_vec = new int[col_len];
  long int *time_vec = new long int[col_len];
  double *lat_vec = new double[col_len];
  double *lng_vec = new double[col_len];

  thrust::copy_n(col_id_vec.begin(),col_len,id_vec);
  thrust::copy_n(col_time_vec.begin(),col_len,time_vec);
  thrust::copy_n(col_lat_vec.begin(),col_len,lat_vec);
  thrust::copy_n(col_lng_vec.begin(),col_len,lng_vec);

  double lat_delta=lat_top-lat_bottom;
  double lng_delta=lng_right-lng_left;
  lat_top+=lat_delta/10;
  lat_bottom-=lat_delta/10;
  lng_right+=lng_delta/10;
  lng_left-=lng_delta/10;
/*
int *seq = new int[col_len];
thrust::copy_n(permutation.begin(),col_len,seq);

for (int i=0;i<col_len;i++)
std::cout<<" lat: "<<lat_vec[i]<<" i: "<<i<<" seq: "<<seq[i]<<" id: "<<id_vec[i]<<std::endl;
*/  
  //data_nodes->clear();
  data_edges->clear();
  int temp_id=-1;
  double last_lat;
  double last_lng;

  //if (col_len>1) data_nodes->push_back(Node(lng_vec[0]*10,lat_vec[0]*10));
  for (int i=0;i<col_len;i++){
    if (time_vec[i]>end_time||time_vec[i]<start_time||lat_vec[i]>lat_top
           ||lat_vec[i]<lat_bottom||lng_vec[i]>lng_right||lng_vec[i]<lng_left
           ){
             temp_id=-1; continue;
           }

    if (id_vec[i]==temp_id&&((lng_vec[i]-last_lng)*(lng_vec[i]-last_lng)
           +(lat_vec[i]-last_lat)*(lat_vec[i]-last_lat)>eps)&&i>0
           &&(time_vec[i]-time_vec[i-1]<MAX_DELTA_TIME)) {
      Node *source = new Node(last_lng*10000,last_lat*10000);
      Node *target = new Node(lng_vec[i]*10000,lat_vec[i]*10000);
      Edge temp_edge(source,target);
      data_edges->push_back(temp_edge);
      last_lat=lat_vec[i];
      last_lng=lng_vec[i];
    } else {
      last_lat=lat_vec[i];
      last_lng=lng_vec[i];
      temp_id=id_vec[i];
    }
    //data_nodes->push_back(Node(lng_vec[i]*10,lat_vec[i]*10));
  }
  /*for (int i=1;i<col_len;i++){
    if (time_vec[i]>end_time||time_vec[i]<start_time) continue;
    std::cout<<"start_lng: "<<lng_vec[i-1]<<" start_lat: "<<lat_vec[i-1]<<" end_lng: "<<lng_vec[i]<<" end_lat: "<<lat_vec[i]<<std::endl;
   Node *source = new Node(lng_vec[i-1]*10000,lat_vec[i-1]*10000);
   Node *target = new Node(lng_vec[i]*10000,lat_vec[i]*10000);
   Edge temp_edge(source,target); 
   data_edges->push_back(temp_edge);
  }*/
};

EdgeBundling::EdgeBundling(
    /*vector<Node> &data_nodes,*/ vector<Edge> &data_edges
    , double K, double S, int P ,int P_rate, int C, int I, double I_rate
    , double compatibility_threshold, bool invers_quadratic_mode)
    : /*data_nodes(data_nodes),*/ data_edges(data_edges), K(K), S(S), P(P)
    , P_rate(P_rate), C(C), I(I), I_rate(I_rate)
    , compatibility_threshold(compatibility_threshold)
    , invers_quadratic_mode(invers_quadratic_mode){}

void EdgeBundling::initializeEdgeSubdivisions(int P_initial){
  subdivision_points_for_edge.clear();
  for (int i=0; i<data_edges.size();i++){
    vector<Node*> temp;
    subdivision_points_for_edge.push_back(temp);
    if (P_initial==1) subdivision_points_for_edge[i].clear();
    else{
      subdivision_points_for_edge[i].clear();
      subdivision_points_for_edge[i].push_back(data_edges[i].source);
      subdivision_points_for_edge[i].push_back(data_edges[i].target);
    }
  }
}

void EdgeBundling::initializeCompatibilityLists(){
  for (int i=0;i<data_edges.size();i++){
    vector<int> temp(0);
    compatibility_list_for_edge.push_back(temp);
    compatibility_list_for_edge[i].clear();
  }
}

void EdgeBundling::computeCompatibilityLists(){
  for (int e=0;e<data_edges.size()-1;e++)
    for (int oe=e+1;oe<data_edges.size();oe++){
      if (areCompatible(data_edges[e],data_edges[oe])){
        compatibility_list_for_edge[e].push_back(oe);
        compatibility_list_for_edge[oe].push_back(e);
      }
    }
}

void EdgeBundling::updateEdgeDivisions(int P){
  for (int e_idx=0;e_idx<data_edges.size();e_idx++){
    if (P==1){
      subdivision_points_for_edge[e_idx].push_back(data_edges[e_idx].source);
      subdivision_points_for_edge[e_idx].push_back(
          edgeMidpoint(data_edges[e_idx]));
      subdivision_points_for_edge[e_idx].push_back(data_edges[e_idx].target);
    } else {
      double divided_edge_length=computeDividedEdgeLength(e_idx);
      double segment_length=divided_edge_length/((P+1)*1.0);
      //std::cout<<"segment length: "<<segment_length<<std::endl;
      if (segment_length<0.0001){
        int pause;
        std::cout<<"e_idx: "<<e_idx<<std::endl;
        std::cin>>pause;
        for (int k=0;k<subdivision_points_for_edge.size();k++){
          std::cout<<"index: "<<k<<" x: "<<subdivision_points_for_edge[e_idx][k]->x<<" y: "<<subdivision_points_for_edge[e_idx][k]->y<<std::endl;
        }
        std::cin>>pause;
      }
      double current_segment_length=segment_length;
      vector<Node*> new_subdivision_points;
      new_subdivision_points.clear();
      new_subdivision_points.push_back(data_edges[e_idx].source);

      for (int i=1; i<subdivision_points_for_edge[e_idx].size();i++){
        double old_segment_length=euclideanDistance(
                   subdivision_points_for_edge[e_idx][i]
                   , subdivision_points_for_edge[e_idx][i-1]);

        while (old_segment_length>current_segment_length+eps){
          double percent_position=current_segment_length/old_segment_length;
          double new_subdivision_point_x
                     =subdivision_points_for_edge[e_idx][i-1]->x;
          double new_subdivision_point_y
                     =subdivision_points_for_edge[e_idx][i-1]->y;
          new_subdivision_point_x+=percent_position
              *(subdivision_points_for_edge[e_idx][i]->x
                   -subdivision_points_for_edge[e_idx][i-1]->x);
          new_subdivision_point_y+=percent_position
              *(subdivision_points_for_edge[e_idx][i]->y
                   -subdivision_points_for_edge[e_idx][i-1]->y);
          Node *temp = new Node(new_subdivision_point_x
                          , new_subdivision_point_y);
          new_subdivision_points.push_back(temp);

          old_segment_length-=current_segment_length;
          current_segment_length=segment_length;
        }
        current_segment_length-=old_segment_length;
      }
      new_subdivision_points.push_back(data_edges[e_idx].target);
      subdivision_points_for_edge[e_idx]=new_subdivision_points;
    }
  }
}

double EdgeBundling::vectorDotProduct(const Node *p, const Node *q){
  return p->x*q->x+p->y*q->y;
}

Node *EdgeBundling::edgeAsVector(const Edge &P){
  Node *result = new Node(P.target->x-P.source->x,P.target->y-P.source->y);
  return result;
}

double EdgeBundling::edgeLength(const Edge &e){
  if (fabs(e.source->x-e.target->x)<eps&&fabs(e.source->y-e.target->y)<eps)
    return eps;
  return sqrt(pow(e.source->x-e.target->x,2)+pow(e.source->y-e.target->y,2));
}

double EdgeBundling::customEdgeLength(const Edge &e){
  return sqrt(pow(e.source->x-e.target->x,2)+pow(e.source->y-e.target->y,2));
}

Node *EdgeBundling::edgeMidpoint(const Edge &e){
  Node *result = new Node((e.source->x+e.target->x)/2.0
      ,(e.source->y+e.target->y)/2.0);
  return result;
}

double EdgeBundling::computeDividedEdgeLength(int e_idx){
  double length=0;
  for (int i=1;i<subdivision_points_for_edge[e_idx].size();i++){
    double segment_length=euclideanDistance(
            subdivision_points_for_edge[e_idx][i]
            , subdivision_points_for_edge[e_idx][i-1]);
    length+=segment_length;
  }
  return length;
}

double EdgeBundling::euclideanDistance(const Node *p,const Node *q){
  return sqrt(pow(p->x-q->x,2)+pow(p->y-q->y,2));
}

Node *EdgeBundling::projectPointOnLine(const Node *p, const Edge &Q){
  double L_squared=(Q.target->x-Q.source->x)*(Q.target->x-Q.source->x)
             +(Q.target->y-Q.source->y)*(Q.target->y-Q.source->y);
  double r=((Q.source->y-p->y)*(Q.source->y-Q.target->y)
             -(Q.source->x-p->x)*(Q.target->x-Q.source->x))/L_squared;
  Node *result = new Node(Q.source->x+r*(Q.target->x-Q.source->x)
      ,Q.source->y+r*(Q.target->y-Q.source->y));
  return result;
}

Node *EdgeBundling::applySpringForce(int e_idx, int i, double kP){
  Node *prev = subdivision_points_for_edge[e_idx][i-1];
  Node *next = subdivision_points_for_edge[e_idx][i+1];
  Node *crnt = subdivision_points_for_edge[e_idx][i];
  double x=prev->x-crnt->x+next->x-crnt->x;
  double y=prev->y-crnt->y+next->y-crnt->y;

  x*=kP;
  y*=kP;
  Node *result = new Node(x,y);
  return result;
}

Node *EdgeBundling::applyElectrostaticForce(int e_idx, int i){
  Node *sum_of_forces = new Node(0,0);
  vector<int> compatible_edges_list(compatibility_list_for_edge[e_idx]);
 
  for (int oe=0; oe<compatible_edges_list.size(); oe++){
    Node *force = new Node(
             subdivision_points_for_edge[compatible_edges_list[oe]][i]->x
             -subdivision_points_for_edge[e_idx][i]->x
             ,subdivision_points_for_edge[compatible_edges_list[oe]][i]->y
             -subdivision_points_for_edge[e_idx][i]->y);

    double compatibility=compatibilityScore(
        data_edges[compatible_edges_list[oe]],data_edges[e_idx]);
    if (fabs(force->x)>eps||fabs(force->y)>eps){
      Edge temp_edge(
          subdivision_points_for_edge[compatible_edges_list[oe]][i]
          , subdivision_points_for_edge[e_idx][i]);
      double diff=(compatibility/pow(customEdgeLength(temp_edge),1));
      sum_of_forces->x+=force->x*diff;
      sum_of_forces->y+=force->y*diff;
    }
  }
  return sum_of_forces;
}

vector<Node*> EdgeBundling::applyResultForcesOnSubdivisionPoints(
    int e_idx, int P, double S){
  double kP=K/(edgeLength(data_edges[e_idx])*(P+1));
  vector<Node*> resulting_forces_for_subdivision_points;
  resulting_forces_for_subdivision_points.clear();
  Node *start_node = new Node(0,0);
  resulting_forces_for_subdivision_points.push_back(start_node);
  for (int i=1;i<P+1;i++){
    Node *resulting_force = new Node(0,0);
    Node *spring_force=applySpringForce(e_idx,i,kP);
    Node *electrostatic_force=applyElectrostaticForce(e_idx,i);

    resulting_force->x=S*(spring_force->x+electrostatic_force->x);
    resulting_force->y=S*(spring_force->y+electrostatic_force->y);

    resulting_forces_for_subdivision_points.push_back(resulting_force);
  }
  Node *end_node = new Node(0,0);
  resulting_forces_for_subdivision_points.push_back(end_node);
  
  return resulting_forces_for_subdivision_points;
}

double EdgeBundling::angleCompatibility(const Edge &P, const Edge &Q){
  return fabs(vectorDotProduct(edgeAsVector(P),edgeAsVector(Q))
             /(edgeLength(P)*edgeLength(Q)));
}

double EdgeBundling::scaleCompatibility(const Edge &P, const Edge &Q){
  double lavg=(edgeLength(P)+edgeLength(Q))/2.0;
  return 2.0/(lavg/std::min(edgeLength(P),edgeLength(Q))
             +max(edgeLength(P),edgeLength(Q))/lavg);
}

double EdgeBundling::positionCompatibility(const Edge &P, const Edge &Q){
  double lavg=(edgeLength(P)+edgeLength(Q));
  Node *midP = new Node((P.source->x+P.target->x)/2.0
      ,(P.source->y+P.target->y)/2.0);
  Node *midQ = new Node((Q.source->x+Q.target->x)/2.0
      ,(Q.source->y+Q.target->y)/2.0);
  return lavg/(lavg+euclideanDistance(midP,midQ));
}

double EdgeBundling::edgeVisibility(const Edge &P, const Edge &Q){
  Node *I0 = projectPointOnLine(Q.source,P);
  Node *I1 = projectPointOnLine(Q.target,P);
  Node *midI = new Node((I0->x+I1->x)/2.0,(I0->y+I1->y)/2.0);
  Node *midP = new Node((P.source->x+P.target->x)/2.0
      ,(P.source->y+P.target->y)/2.0);
  return std::max(0.0,1-2*euclideanDistance(midP,midI)
             /euclideanDistance(I0,I1));
}

double EdgeBundling::visibilityCompatibility(const Edge &P, const Edge &Q){
  return std::min(edgeVisibility(P,Q),edgeVisibility(Q,P));
}

double EdgeBundling::edgeProximity(const Edge &P, const Edge &Q){
  double proximity_source=fabs(P.source->x-Q.source->x)
             +fabs(P.source->y-Q.source->y);
  double proximity_target=fabs(P.target->x-Q.target->x)
             +fabs(P.target->y-Q.target->y);
  return std::max(0.0,1-(proximity_source+proximity_target)
             /(edgeLength(P)+edgeLength(Q)));
}

double EdgeBundling::proximityCompatibility(const Edge &P, const Edge &Q){
  return std::min(edgeProximity(P,Q),edgeProximity(Q,P));
}

double EdgeBundling::compatibilityScore(const Edge &P, const Edge &Q){
  return (angleCompatibility(P,Q)*scaleCompatibility(P,Q)
             *positionCompatibility(P,Q)*visibilityCompatibility(P,Q));
}

bool EdgeBundling::areCompatible(const Edge &P, const Edge &Q){
  return (compatibilityScore(P,Q)>=compatibility_threshold);
}

void EdgeBundling::forceBundle(){
  if (data_edges.size()>9000){
    subdivision_points_for_edge.clear();
    return;
  }
  std::cout<<"1"<<std::endl;
  initializeEdgeSubdivisions(P);
  std::cout<<"2"<<std::endl;
  initializeCompatibilityLists();
  std::cout<<"3"<<std::endl;
  updateEdgeDivisions(P);
  std::cout<<"edge_size"<<data_edges.size()<<std::endl;
  computeCompatibilityLists();
  std::cout<<"initialization finished"<<std::endl;
  
  for (int cycle=0;cycle<C;cycle++){
    for (int iteration=0;iteration<I;iteration++){
      vector<vector<Node*>> forces;
      forces.clear();
      for (int edge=0;edge<data_edges.size();edge++){
        forces.push_back(applyResultForcesOnSubdivisionPoints(edge,P,S));
      }
      for (int e=0;e<data_edges.size();e++){
        //std::cerr<<subdivision_points_for_edge[e].size()<<" P: "<<P<<std::endl;
        //for (int i=0;i<forces[e].size();i++)
        //  std::cerr<<"i: "<<i<<"force x: "<<forces[e][i]->x<<"\t";

        for (int i=0;i<P+1;i++){
          subdivision_points_for_edge[e][i]->x+=forces[e][i]->x;
          subdivision_points_for_edge[e][i]->y+=forces[e][i]->y;
        }
      }
    }
    S=S/2;
    P=P*P_rate;
    I=I_rate*I;
    updateEdgeDivisions(P);
  }
}

}  //edgebundling
}  //selika
