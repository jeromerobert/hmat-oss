#pragma once

#include <vector>
#include <complex>
#include "cluster_assembly_function.hpp"
#include "data_types.hpp"
#include "scalar_array.hpp"

namespace hmat {

  template<typename T>
  struct Pivot {
    int row_;
    int col_;
    T value_;

    Pivot(int row, int col, T value) : row_(row), col_(col), value_(value) {}
    Pivot() : row_(0), col_(0), value_(0) {}
    static bool ComparerLower(const Pivot& pivot1, const Pivot& pivot2){
      return std::abs(pivot1.value_) > std::abs(pivot2.value_);
    }
  };

  template<typename T>
  class RandomPivotManager {
    typedef typename Types<T>::dp dp_t;
    const hmat::ClusterAssemblyFunction<T> &clusterAssemblyFunction_;
    std::vector<Pivot<dp_t> > pivots_;
    double refValue_;
    int usedPivots_;

  public:
    RandomPivotManager(const hmat::ClusterAssemblyFunction<T> &function, int nSamples);

    void AddUsedPivot(Vector<dp_t> *row, Vector<dp_t> *col, int rowIndex, int colIndex);

    Pivot<dp_t> GetPivot();

  };
}
