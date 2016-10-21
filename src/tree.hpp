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
#include <assert.h>

namespace hmat {

// Forward declaration
template <typename TreeNode> class Tree;

/* Visitor pattern
 */
enum Visit { tree_preorder, tree_postorder, tree_inorder, tree_leaf };

/** Class to recursively apply a given function to all nodes of a tree
 */
template <typename TreeNode>
class TreeProcedure {

public:
  TreeProcedure() {}
  virtual void visit(TreeNode* node, const Visit order) const = 0;
  virtual ~TreeProcedure() {}
};

/*! \brief Templated tree class.

  This class represents a tree of arity N, holding an instance of NodeData in
  its nodes.
 */
template <typename TreeNode>
class Tree {
public:
  /// depth of the current node in the tree
  int depth;

protected:
  /// empty for a leaf, pointer on a vector of sons otherwise.
  std::vector<TreeNode*> children;
public:
  /// Pointer to the father, NULL if this node is the root
  TreeNode* father;

public:
  Tree(TreeNode* _father, int _depth = 0)
    : depth(_depth), children(), father(_father) {}
  virtual ~Tree() {
    for (int i=0 ; i<nrChild() ; i++)
      if (children[i])
        delete children[i];
    children.clear();
  }

  // https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
  // "me()->" replaces "this->" when calling a method of TreeNode
  TreeNode* me() {
      return static_cast<TreeNode*>(this);
  }
  const TreeNode* me() const {
      return static_cast<const TreeNode*>(this);
  }

  /*! \brief Insert a child in the children vector.

    If a child is already present, it is removed but not deleted.

    \param index index in the children vector
    \param child pointeur to the child
   */
  void insertChild(int index, TreeNode *child) {
    if (nrChild()<=index)
      children.resize(index+1, (TreeNode*)NULL);
    children[index] = child;
    if (child) {
      child->father = me();
      child->depth = depth + 1;
    }
  }

  /*! \brief Remove a child, and delete it if necessary.
   */
  void removeChild(int index) {
    assert(index>=0 && index<nrChild());
    if (children[index])
    delete children[index];
    children[index] = (TreeNode*)NULL;
  }

  /*! \brief Return the number of nodes in the tree.
   */
  int nodesCount() const {
    int result = 1;
    for (int i=0 ; i<nrChild() ; i++)
      if (children[i])
        result += children[i]->nodesCount();
    return result;
  }

  /*! \brief Return the child of index, or NULL.
   */
  inline TreeNode *getChild(int index) const {
    assert(index>=0 && index<nrChild());
    return children[index];
  }
  inline TreeNode *&getChild(int index)  {
    assert(index>=0 && index<nrChild());
    return children[index];
  }

  inline TreeNode *getFather() const {
    return father;
  }

  inline int nrChild() const {
    return (int)children.size();
  }

  /*! \brief Return true if the node is a leaf (= it has no children).
   */
  inline bool isLeaf() const {
    return children.empty();
  }

  /*! \brief Return a list of nodes.

    Not used anywhere.
   */
  virtual std::list<const TreeNode*> listNodes() const {
    std::list<const TreeNode*> result;
    result.push_back(me());
    for (int i=0 ; i<nrChild() ; i++)
      if (children[i]) {
        std::list<const TreeNode*> childNodes = children[i]->listNodes();
        result.splice(result.end(), childNodes, childNodes.begin(), childNodes.end());
      }
    return result;
  }

 /*! \brief Return a list of leaves.
   */
  void listAllLeaves(std::vector<const TreeNode*>& leaves) const {
    if (!isLeaf()) {
      for (int i=0 ; i<nrChild() ; i++)
        if (children[i])
          children[i]->listAllLeaves(leaves);
    } else {
      leaves.push_back(me());
    }
  }

  void walk(const TreeProcedure<TreeNode> *proc) {
    if (isLeaf()) {
      proc->visit(me(), tree_leaf); // treatment on the leaves
    } else {
      proc->visit(me(), tree_preorder); // treatment on a non-leaf before recursion
      bool first = true;
      for (int i=0 ; i<nrChild() ; i++)
        if (children[i]) {
          if (!first)
            proc->visit(me(), tree_inorder); // treatment on a non-leaf after 1st child (mainly usefull with 2 children)
          first = false;
          children[i]->walk(proc);
        }
      proc->visit(me(), tree_postorder); // treatment on a non-leaf after recursion
    }
  }

};

}  // end namespace hmat

#endif  // _TREE_HPP
