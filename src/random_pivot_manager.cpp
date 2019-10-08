#include <cstdlib>
#include <algorithm>
#include "data_types.hpp"
#include "random_pivot_manager.hpp"

namespace hmat {

  template<typename T>
  RandomPivotManager<T>::RandomPivotManager(const ClusterAssemblyFunction<T> &function, int nSamples):
      _clusterAssemblyFunction(function) {

    if(nSamples==0)
      return;

    int nRows = _clusterAssemblyFunction.rows->size();
    int nCols = _clusterAssemblyFunction.cols->size();

    for (int i = 0; i < nSamples; ++i) {
      int row = rand() % nRows;
      int col = rand() % nCols;
      dp_t value = function.getElement(row, col);
      _pivots.push_back(Pivot<dp_t>(row, col, value));
    }

    std::sort(_pivots.begin(), _pivots.end(), Pivot<dp_t>::ComparerLower);
    _refValue = sqrt(squaredNorm(_pivots[0].value));
  }

  template<typename T>
  void RandomPivotManager<T>::AddUsedPivot(Vector<dp_t> *row, Vector<dp_t> *col, int rowIndex, int colIndex) {
    _usedPivots +=1;
    if (_pivots.empty())
      return;

    int numberOfPivotsToRemove = 0;

    for (int i = 0; i < _pivots.size(); ++i) {
      Pivot<dp_t> &pivot = _pivots[i];
      pivot.value -= (*row)[pivot.col] * (*col)[pivot.row];
      if (pivot.row == rowIndex || pivot.col == colIndex)
        numberOfPivotsToRemove += 1;
    }
    std::sort(_pivots.begin(), _pivots.end(), Pivot<dp_t>::ComparerLower);
    int size;
    for (size = _pivots.size() - 1; size >= 0; --size) {
      if (abs(_pivots[size].value) > 1e-14 * _refValue)
        break;
    }
    assert(_pivots.size() - (size + 1) >= numberOfPivotsToRemove);
    _pivots.resize(size + 1);
  }

  template<typename T>
  Pivot<typename Types<T>::dp> RandomPivotManager<T>::GetPivot() {
    if (_pivots.empty())
      return Pivot<dp_t>();
    return _pivots[0];
  }

  // Declaration of the used templates
  template
  class RandomPivotManager<S_t>;

  template
  class RandomPivotManager<D_t>;

  template
  class RandomPivotManager<C_t>;

  template
  class RandomPivotManager<Z_t>;

}