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

#include "hmat/config.h"

#include "context.hpp"

#include <assert.h>
#include <cstring>
#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

namespace trace {

  int (*nodeIndexFunction)() = NULL;

  /** \brief Set the function used to get the root index.

      \return a number between 0 (not in a parallel region) and the number of
      workers (included).
   */
  static int currentNodeIndex() {
    int res = (nodeIndexFunction ? nodeIndexFunction() : -1) + 1;
    assert(res>=0 && res<MAX_ROOTS);
    return res;
  }

  void setNodeIndexFunction(int (*nodeIndexFunc)()) {
    nodeIndexFunction = nodeIndexFunc;
  }

  bool Node::enabled = true;
  UM_NS::unordered_map<void*, Node*> Node::currentNodes[MAX_ROOTS];
  void* Node::enclosingContext[MAX_ROOTS] = {};

  Node::Node(const char* _name, Node* _parent)
    : name(_name), data(), parent(_parent), children() {}

  Node::~Node() {
    for (std::vector<Node*>::iterator it = children.begin(); it != children.end(); ++it) {
      delete *it;
    }
  }

  void Node::enterContext(const char* name) {
    Node* current = currentNode();
    assert(current);
    Node* child = current->findChild(name);
    int index = currentNodeIndex();
    void* enclosing = enclosingContext[index];

    if (!child) {
      child = new Node(name, current);
      current->children.push_back(child);
    }
    assert(child);
    currentNodes[index][enclosing] = child;
    current = child;
    current->data.lastEnterTime = now();
    current->data.n += 1;
  }

  void Node::leaveContext() {
    int index = currentNodeIndex();
    void* enclosing = enclosingContext[index];
    Node* current = currentNodes[index][enclosing];
    assert(current);

    current->data.totalTime += time_diff_in_nanos(current->data.lastEnterTime, now());

    if (!(current->parent)) {
      std::cout << "Warning! Closing root node." << std::endl;
    } else {
      currentNodes[index][enclosing] = current->parent;
    }
  }

  void* Node::currentReference() {
    return (void*) Node::currentNode();
  }

  void Node::setEnclosingContext(void* enclosing) {
    int index = currentNodeIndex();
    enclosingContext[index] = enclosing;
  }

  void Node::incrementFlops(int64_t flops) {
    currentNode()->data.totalFlops += flops;
  }

  void Node::startComm() {
    currentNode()->data.lastCommInitiationTime = now();
  }

  void Node::endComm() {
    Node* current = currentNode();
    current->data.totalCommTime += time_diff_in_nanos(current->data.lastEnterTime, now());
  }

  Node* Node::findChild(const char* name) const {
    for (std::vector<Node*>::const_iterator it = children.begin(); it != children.end(); ++it) {
      // On cherche la correspondance avec le pointeur. Puisqu'on demande que
      // tous les noms soient des pointeurs qui existent tout le long de
      // l'execution, on peut forcer l'unicite.
      if ((*it)->name == name) {
	return *it;
      }
    }
    return NULL;
  }

  void Node::dump(std::ofstream& f) const {
    f << "{"
      << "\"name\": \"" << name << "\", "
      << "\"id\": \"" << this << "\", "
      << "\"n\": " << data.n << ", "
      << "\"totalTime\": " << data.totalTime / 1e9 << ", "
      << "\"totalFlops\": " << data.totalFlops << ", "
      << "\"totalBytesSent\": " << data.totalBytesSent << ", "
      << "\"totalBytesReceived\": " << data.totalBytesReceived << ", "
      << "\"totalBytesReceived\": " << data.totalBytesReceived << ", "
      << "\"totalCommTime\": " << data.totalCommTime / 1e9 << "," << std::endl;
    f << "\"children\": [";
    std::string delimiter("");
    for (std::vector<Node*>::const_iterator it = children.begin(); it != children.end(); ++it) {
      f << delimiter;
      (*it)->dump(f);
      delimiter = ", ";
    }
    f << "]}";
  }

  void Node::jsonDump(const char* filename) {
    std::ofstream f(filename);

    f << "[";
    std::string delimiter("");
    for (int i = 0; i < MAX_ROOTS; i++) {
      if (!currentNodes[i].empty()) {
        UM_NS::unordered_map<void*, Node*>::iterator p = currentNodes[i].begin();
        for(; p != currentNodes[i].end(); ++p) {
          Node* root = p->second;
          f << delimiter << std::endl;
          root->dump(f);
          delimiter = ", ";
        }
      }
    }
    f << std::endl << "]" << std::endl;
  }

  /** Find the current node, allocating one if necessary.
   */
  Node* Node::currentNode() {
    int index = currentNodeIndex();
    void* enclosing = enclosingContext[index];
    UM_NS::unordered_map<void*, Node*>::iterator it = currentNodes[index].find(enclosing);
    Node* current;
    if (it == currentNodes[index].end()) {
      char *name = const_cast<char*>("root");
      if (index != 0) {
        name = strdup("Worker #XXX - 0xXXXXXXXXXXXXXXXX"); // Worker ID - enclosing
        assert(name);
        sprintf(name, "Worker #%03d - %p", index, enclosing);
      }
      current = new Node(name, NULL);
      currentNodes[index][enclosing] = current;
    } else {
      current = it->second;
    }
    return current;
  }
}
