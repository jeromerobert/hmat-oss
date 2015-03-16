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

/*! \file
  \ingroup HMatrix
  \brief Templated Tree class used by ClusterTree and HMatrix.
*/
#ifndef _TREE_HPP
#define _TREE_HPP
#include <vector>
#include <list>
#include <cstddef>

/*! \brief Templated tree class.

  This class represents a tree of arity N, holding an instance of NodeData in
  its nodes.
 */
template<int N> class Tree {
public:
  /// depth of the current node in the tree
  int depth;

protected:
  /// NULL for a leaf, pointeur on an array of N sons otherwise.
  Tree** children;
public:
  /// Pointer to the father, NULL if this node is the root
  Tree* father;

public:
  Tree(Tree* _father, int _depth = 0)
    : depth(_depth), children(NULL), father(_father) {}
  virtual ~Tree() {
    if (!children) {
      return;
    }
    for (int i = 0; i < N; i++) {
      if (children[i]) {
        delete children[i];
        children[i] = NULL;
      }
    }
    delete[] children;
  }

  /*! \brief Insert a child in the children array.

    If a child is already present, it is removed but not deleted.

    \param index index in the children array
    \param child pointeur to the child
   */
  void insertChild(int index, Tree *child) {
    if (!children) {
      children = new Tree*[N];
      for (int i = 0; i < N; i++) {
        children[i] = NULL;
      }
    }
    child->father = this;
    children[index] = child;
    child->depth = depth + 1;
  }

  /*! \brief Remove a children, and delete it if necessary.
   */
  void removeChild(int index) {
    delete children[index];
    children[index] = NULL;
  }

  /*! \brief Return the number of nodes in the tree.
   */
  int nodesCount() const {
    int result = 1;
    if (!isLeaf()) {
      for (int i = 0; i < N; i++) {
        if (getChild(i)) {
          result += getChild(i)->nodesCount();
        }
      }
    }
    return result;
  }

  /*! \brief Return the child of index, or NULL.

    \warning Will segfault if used on a leaf.
   */
  inline Tree *getChild(int index) const {
    return children[index];
  }

  /*! \brief Return true if the node is a leaf.
   */
  inline bool isLeaf() const {
    return !children;
  }

  /*! \brief Return a list of nodes.
   */
  virtual std::list<const Tree<N>*> listNodes() const {
    std::list<const Tree<N>*> result;
    result.push_back(this);
    if (!isLeaf()) {
      for (int i = 0; i < N; i++) {
        Tree<N>* child = getChild(i);
        if (child) {
          std::list<const Tree<N>*> childNodes = child->listNodes();
          result.splice(result.end(), childNodes, childNodes.begin(),
                        childNodes.end());
        }
      }
    }
    return result;
  }

 /*! \brief Return a list of leaves.
   */
  void listAllLeaves(std::vector<Tree<N>*>& leaves) const {
    if (!isLeaf()) {
      for (int i = 0; i < N; i++) {
        Tree<N>* child = getChild(i);
        if (child) {
          child->listAllLeaves(leaves);
        }
      }
    } else {
      leaves.push_back(const_cast<Tree<N>*>(this));
    }
  }
};
#endif  // _TREE_HPP
