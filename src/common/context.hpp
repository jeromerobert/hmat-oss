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
  \brief Context manager used for the tracing functionnality.
*/
#ifndef _CONTEXT_HPP
#define _CONTEXT_HPP

#include "hmat/config.h"
#include "common/chrono.h"
#include <vector>
#include <fstream>

#if (__cplusplus > 201103L) || defined(HAVE_CPP11) || defined(_MSC_VER) || defined(_LIBCPP_VERSION)
  #include <unordered_map>
  #define UM_NS std
#else
  #include <tr1/unordered_map>
  #define UM_NS std::tr1
#endif

namespace hmat {

// See http://herbsutter.com/2009/10/18/mailbag-shutting-up-compiler-warnings/
template<class T>
void ignore_unused_arg( const T& ) {}

}

namespace trace {

  class NodeData {
  public:
    unsigned int n;
    int64_t totalTime; // ns
    int64_t totalFlops;
    int totalBytesSent;
    int totalBytesReceived;
    int64_t totalCommTime;
    Time lastEnterTime;
    Time lastCommInitiationTime;
  };

  // Maximum number of parallel workers + 1 (for the main non-parallel context)
#ifndef MAX_ROOTS
  #define MAX_ROOTS 128
#endif

  /** Set the function tasked with returning the current root index.

      The function set this way is used to determine to which tree the current
      tracing context belongs to. By default, a single context is assumed, and
      this behavior can be restored by setting the fuction to NULL.

      \param nodeIndexFunc a function returning a worker index, which is:
        - 0 in non-parallel parts of the code
        - between 1 and n_workers (included) in parallel regions
   */
  void setNodeIndexFunction(int (*nodeIndexFunc)());

  class Node {
  public:
    /// True if the tracing is enabled. True by default. Not used anywhere, apparently...
    static bool enabled;
    static const char* flops_count_filename;
  private:
    /// Unique name for the context.
    const char* name_;
    /// Tracing data associated with this node.
    NodeData data;
    /// Parent node. NULL for a root.
    Node* parent;
    /// Ordered list of children nodes.
    std::vector<Node*> children;
    /// List of trace trees.
    static UM_NS::unordered_map<void*, Node*> currentNodes[MAX_ROOTS]; // TODO: padding to avoid false sharing ?
    static void* enclosingContext[MAX_ROOTS]; // TODO: padding to avoid false sharing ?


  public:
    /** Enter a context noted by a name.
     */
    static void enterContext(const char* name_);
    /** Leave the current context.
     */
    static void leaveContext();
    /** Get a unique reference to the current context. */
    static void* currentReference();
    /** Set the current context as being relative to a given enclosing one.

        The enclosing context is identified through the pointer returned by \a
        currentReference().
     */
    static void setEnclosingContext(void* enclosing);
    static void enable() {enabled = true;}
    static void disable() {enabled = false;}
    static void incrementFlops(int64_t flops);
    static void startComm();
    static void endComm();
    /** Dumps the trace trees to a JSON file.
     */
    static void jsonDumpMain(const char* filename);

  private:
    Node(const char* _name, Node* _parent);
    ~Node();
    Node* findChild(const char* name) const;
    void jsonDump(std::ofstream& f) const;
    static Node* currentNode();
  };
}


class DisableContextInBlock {
private:
  bool enabled;
public:
  DisableContextInBlock() {
    enabled = trace::Node::enabled;
    trace::Node::disable();
  }
  ~DisableContextInBlock() {
    trace::Node::enabled = enabled;
  }
};

#ifdef HAVE_CONTEXT

// Since 'enable' is not used anywhere, this macro probably does nothing.
#define DISABLE_CONTEXT_IN_BLOCK DisableContextInBlock dummyDisableContextInBlock

#define tracing_set_worker_index_func(f) trace::setNodeIndexFunction(f)
#define enter_context(x) trace::Node::enterContext(x)
#define leave_context() trace::Node::leaveContext()
#define increment_flops(x) trace::Node::incrementFlops(x)
#define tracing_dump(x) trace::Node::jsonDumpMain(x)

#else
#define tracing_set_worker_index_func(f) do {} while (0)
#define enter_context(x) do { hmat::ignore_unused_arg(x); } while(0)
#define leave_context()  do {} while(0)
#define increment_flops(x) do { hmat::ignore_unused_arg(x); } while(0)
#define tracing_dump(x) do { hmat::ignore_unused_arg(x); } while(0)
#define DISABLE_CONTEXT_IN_BLOCK do {} while (0)
#endif


/*! \brief Simple wrapper around enter/leave_context() to avoid
having to put leave_context() before each return statement. */
class Context {
public:
  Context(const char* name) {
    enter_context(name);
  }
  ~Context() {
    leave_context();
  }
};

#if defined(__GNUC__)
#define DECLARE_CONTEXT Context __reserved_ctx(__PRETTY_FUNCTION__)
#elif defined(_MSC_VER)
#define DECLARE_CONTEXT Context __reserved_ctx(__FUNCTION__)
#else
#define DECLARE_CONTEXT Context __reserved_ctx(__func__)
#endif

#endif
