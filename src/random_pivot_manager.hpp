#pragma once

#include <vector>
#include <complex>
#include "cluster_assembly_function.hpp"
#include "data_types.hpp"
#include "scalar_array.hpp"

namespace hmat {

  template<typename T>
  struct Pivot {
    int row;
    int col;
    T value;

    Pivot(int row, int col, T value) : row(row), col(col), value(value) {}
    Pivot() : row(0), col(0), value(0) {}
    static bool ComparerLower(const Pivot& pivot1, const Pivot& pivot2){ return abs(pivot1.value) > abs(pivot2.value);}
  };

  template<typename T>
  class RandomPivotManager {
    typedef typename Types<T>::dp dp_t;
    const hmat::ClusterAssemblyFunction<T> &_clusterAssemblyFunction;
    std::vector<Pivot<dp_t> > _pivots;
    double _refValue;
    int _usedPivots;

  public:
    RandomPivotManager(const hmat::ClusterAssemblyFunction<T> &function, int nSamples);

    void AddUsedPivot(Vector<dp_t> *row, Vector<dp_t> *col, int rowIndex, int colIndex);

    Pivot<dp_t> GetPivot();

  };
}