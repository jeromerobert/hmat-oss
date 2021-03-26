#include <cstdlib>
#include <algorithm>
#include "data_types.hpp"
#include "random_pivot_manager.hpp"

namespace hmat {

  template<typename T>
  RandomPivotManager<T>::RandomPivotManager(const ClusterAssemblyFunction<T> &function, int nSamples):
      clusterAssemblyFunction_(function) {

    if(nSamples==0)
      return;

    int nRows = clusterAssemblyFunction_.rows->size();
    int nCols = clusterAssemblyFunction_.cols->size();

    for (int i = 0; i < nSamples; ++i) {
      int row = rand() % nRows;
      int col = rand() % nCols;
      dp_t value = function.getElement(row, col);
      pivots_.push_back(Pivot<dp_t>(row, col, value));
    }

    std::sort(pivots_.begin(), pivots_.end(), Pivot<dp_t>::ComparerLower);
    refValue_ = sqrt(squaredNorm(pivots_[0].value_));
  }

  template<typename T>
  void RandomPivotManager<T>::AddUsedPivot(Vector<dp_t> *row, Vector<dp_t> *col, int rowIndex, int colIndex) {
    usedPivots_ +=1;
    if (pivots_.empty())
      return;

    int numberOfPivotsToRemove = 0;

    for (int i = 0; i < pivots_.size(); ++i) {
      Pivot<dp_t> &pivot = pivots_[i];
      pivot.value_ -= (*row)[pivot.col_] * (*col)[pivot.row_];
      if (pivot.row_ == rowIndex || pivot.col_ == colIndex)
        numberOfPivotsToRemove += 1;
    }
    std::sort(pivots_.begin(), pivots_.end(), Pivot<dp_t>::ComparerLower);
    int size;
    for (size = pivots_.size() - 1; size >= 0; --size) {
      if (std::abs(pivots_[size].value_) > 1e-14 * refValue_)
        break;
    }
    assert(pivots_.size() - (size + 1) >= numberOfPivotsToRemove);
    pivots_.resize(size + 1);
  }

  template<typename T>
  Pivot<typename Types<T>::dp> RandomPivotManager<T>::GetPivot() {
    if (pivots_.empty())
      return Pivot<dp_t>();
    return pivots_[0];
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
