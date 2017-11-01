#ifndef LEXICOGRAPHICAL_SORT_H__
#define LEXICOGRAPHICAL_SORT_H__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

template <typename KeyVector, typename PermutationVector>
void update_permutation(KeyVector& keys, PermutationVector& permutation){
  KeyVector temp(keys.size());
  thrust::gather(permutation.begin(),permutation.end()
      ,keys.begin(),temp.begin());
  thrust::stable_sort_by_key(temp.begin(),temp.end(),permutation.begin());
}

template <typename KeyVector, typename PermutationVector>
void apply_permutation(KeyVector& keys,PermutationVector& permutation){
  KeyVector temp(keys.begin(),keys.end());
  thrust::gather(permutation.begin(),permutation.end()
      ,temp.begin(),keys.begin());
}

#endif  //LEXICOGRAPHICAL_SORT_H__
