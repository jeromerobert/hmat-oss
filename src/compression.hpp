/*
  HMat-OSS (HMatrix library, open source software)

  Copyright (C) 2014-2015 Airbus Group SAS

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2
  of the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

  http://github.com/jeromerobert/hmat-oss
*/

#ifndef _COMPRESSION_HPP
#define _COMPRESSION_HPP
/* Implementation of the algorithms of blocks compression */
#include "data_types.hpp"

/** Choice of the compression method.
 */
#include "assembly.hpp"
#include "cluster_tree.hpp"

namespace hmat {

// Forward declarations
template<typename T> class ScalarArray;
template<typename T> class FullMatrix;
template<typename T> class RkMatrix;
template<typename T> class ClusterAssemblyFunction;
template<typename T> class Function;
class ClusterData;

enum CompressionMethod {
  Svd, AcaFull, AcaPartial, AcaPlus, NoCompression, AcaRandom
};
class IndexSet;


/** Compress a FullMatrix into an RkMatrix.

    The compression uses the reduced SVD, and the accuracy is
    controlled by eps.

    \param m The matrix to compress. It is modified but not detroyed by the function.
    \param eps Accuracy
    \return A RkMatrix approximationg the argument \a m.
*/
template<typename T>
RkMatrix<T>* truncatedSvd(FullMatrix<T>* m, double eps);

/** Fast alternative to ScalarArray::truncatedSvdDecomposition based on ACA full */
template<typename T>
void acaFull(ScalarArray<T> & m, ScalarArray<T>* & u, ScalarArray<T>* & v, double eps);

/** Fast alternative to truncatedSvd based on ACA full */
template<typename T>
RkMatrix<T>* acaFull(FullMatrix<T>* m, double eps);

// Abstract class to compress a block into an RkMatrix.
class CompressionAlgorithm
{
public:
    explicit CompressionAlgorithm(double epsilon) : epsilon_(epsilon) {}
    virtual ~CompressionAlgorithm() {}
    virtual CompressionAlgorithm* clone() const = 0;
    // Compress a block into an RkMatrix, block elements are retrieved by ClusterAssemblyFunction when needed
    virtual RkMatrix<Types<S_t>::dp>* compress(const ClusterAssemblyFunction<S_t>& block) const = 0;
    virtual RkMatrix<Types<D_t>::dp>* compress(const ClusterAssemblyFunction<D_t>& block) const = 0;
    virtual RkMatrix<Types<C_t>::dp>* compress(const ClusterAssemblyFunction<C_t>& block) const = 0;
    virtual RkMatrix<Types<Z_t>::dp>* compress(const ClusterAssemblyFunction<Z_t>& block) const = 0;
    // Get threshold
    virtual double getEpsilon() const { return epsilon_; }
    // Tell whether algorithm needs the whole block or works incrementally.
    virtual bool isIncremental(const ClusterData&, const ClusterData&) const { return true; }
protected:
    double epsilon_;
};


class CompressionSVD : public CompressionAlgorithm
{
public:
    explicit CompressionSVD(double epsilon) : CompressionAlgorithm(epsilon) {}
    CompressionSVD* clone() const { return new CompressionSVD(epsilon_); }
    RkMatrix<Types<S_t>::dp>* compress(const ClusterAssemblyFunction<S_t>& block) const;
    RkMatrix<Types<D_t>::dp>* compress(const ClusterAssemblyFunction<D_t>& block) const;
    RkMatrix<Types<C_t>::dp>* compress(const ClusterAssemblyFunction<C_t>& block) const;
    RkMatrix<Types<Z_t>::dp>* compress(const ClusterAssemblyFunction<Z_t>& block) const;
    bool isIncremental(const ClusterData&, const ClusterData&) const { return false; }
};


class CompressionAcaFull : public CompressionAlgorithm
{
public:
    explicit CompressionAcaFull(double epsilon) : CompressionAlgorithm(epsilon) {}
    CompressionAcaFull* clone() const { return new CompressionAcaFull(epsilon_); }
    RkMatrix<Types<S_t>::dp>* compress(const ClusterAssemblyFunction<S_t>& block) const;
    RkMatrix<Types<D_t>::dp>* compress(const ClusterAssemblyFunction<D_t>& block) const;
    RkMatrix<Types<C_t>::dp>* compress(const ClusterAssemblyFunction<C_t>& block) const;
    RkMatrix<Types<Z_t>::dp>* compress(const ClusterAssemblyFunction<Z_t>& block) const;
    bool isIncremental(const ClusterData&, const ClusterData&) const { return false; }
};


class CompressionAcaPartial : public CompressionAlgorithm
{
public:
    explicit CompressionAcaPartial(double epsilon) : CompressionAlgorithm(epsilon), useRandomPivots_(false) {}
    CompressionAcaPartial* clone() const { return new CompressionAcaPartial(epsilon_); }
    RkMatrix<Types<S_t>::dp>* compress(const ClusterAssemblyFunction<S_t>& block) const;
    RkMatrix<Types<D_t>::dp>* compress(const ClusterAssemblyFunction<D_t>& block) const;
    RkMatrix<Types<C_t>::dp>* compress(const ClusterAssemblyFunction<C_t>& block) const;
    RkMatrix<Types<Z_t>::dp>* compress(const ClusterAssemblyFunction<Z_t>& block) const;
protected:
    bool useRandomPivots_;
};


class CompressionAcaPlus : public CompressionAlgorithm
{
public:
    explicit CompressionAcaPlus(double epsilon) : CompressionAlgorithm(epsilon), delegate_(new CompressionAcaPartial(epsilon)) {}
    ~CompressionAcaPlus() { delete delegate_; }
    CompressionAcaPlus* clone() const { return new CompressionAcaPlus(epsilon_); }
    RkMatrix<Types<S_t>::dp>* compress(const ClusterAssemblyFunction<S_t>& block) const;
    RkMatrix<Types<D_t>::dp>* compress(const ClusterAssemblyFunction<D_t>& block) const;
    RkMatrix<Types<C_t>::dp>* compress(const ClusterAssemblyFunction<C_t>& block) const;
    RkMatrix<Types<Z_t>::dp>* compress(const ClusterAssemblyFunction<Z_t>& block) const;
private:
    // ACA+ start with a findMinRow call which will last for hours
    // if the block contains many null rows
    CompressionAcaPartial * delegate_;
};


class CompressionAcaRandom : public CompressionAcaPartial
{
public:
    explicit CompressionAcaRandom(double epsilon) : CompressionAcaPartial(epsilon) { this->useRandomPivots_ = true; }
    CompressionAcaRandom* clone() const { return new CompressionAcaRandom(epsilon_); }
};

class CompressionRRQR : public CompressionAlgorithm
{
    public :
        explicit CompressionRRQR (double epsilon) : CompressionAlgorithm(epsilon){}
        CompressionRRQR* clone() const { return new CompressionRRQR(epsilon_); }
        RkMatrix<Types<S_t>::dp>* compress(const ClusterAssemblyFunction<S_t>& block) const;
        RkMatrix<Types<D_t>::dp>* compress(const ClusterAssemblyFunction<D_t>& block) const;
        RkMatrix<Types<C_t>::dp>* compress(const ClusterAssemblyFunction<C_t>& block) const;
        RkMatrix<Types<Z_t>::dp>* compress(const ClusterAssemblyFunction<Z_t>& block) const;
    
};

template<typename T>
RkMatrix<typename Types<T>::dp>*
compress(const CompressionAlgorithm* compression, const Function<T>& f,
         const ClusterData* rows, const ClusterData* cols, double epsilon,
         const AllocationObserver & = AllocationObserver());

}  // end namespace hmat
#endif
