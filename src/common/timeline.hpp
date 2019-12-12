#pragma once

#include "config.h"
#include "h_matrix.hpp"
#include "common/chrono.h"
#include "common/context.hpp"

namespace hmat {

class Timeline {
#ifdef HMAT_TIMELINE
    std::vector<FILE *> files_;
    bool enabled_;
    bool packEnabled_;
    bool gemmEnabled_;
    bool qrEnabled_;
    bool onlyWorker_;
    Timeline() : enabled_(false), packEnabled_(true), gemmEnabled_(false), qrEnabled_(false) {}
    ~Timeline();
#endif
    public:
    enum Operation { GEMM, AXPY, SOLVE_UPPER, LLT, LDLT, MDMT, M_DIAG,
                     SOLVE_UPPER_LEFT, ASM, ASM_SYM, SOLVE_LOWER_LEFT,
                     PACK, UNPACK, INIT, PACK_COUNT, EXTRACT_RK, ASSEMBLE_RK,
                     MGS, QR, BLASGEMM, COPY_TRUNCATE, COARSEN};
    class Task {
#ifdef HMAT_TIMELINE
        char buffer[68]; // 68 bytes = 'op' (4 bytes) + payload (48 bytes max) + 2 timestamps (2x8 bytes)
        int buffer_size;
        Timeline & timeline_;
        int workerId_;
        template<typename T> void write(T num) {
            *reinterpret_cast<T*>(buffer+buffer_size) = num;
            buffer_size += sizeof(T);
        }
        bool init(int workerId, Operation op);
        void timestamp();
        void addBlock(const IndexSet * rows, const IndexSet * cols);
        template<typename T> void addBlock(HMatrix<T> * m) {
            addBlock(m->rows(), m->cols());
        }
    public:
        /** Shortcut API to record a new Task with blocks */
        template<typename T> Task(Operation op, HMatrix<T> * block1,
            HMatrix<T> * block2 = NULL, HMatrix<T> * block3 = NULL):
            buffer_size(0), timeline_(instance()) {
          // I use trace::currentNodeIndex() to get the worker id for any runtime.
          // I do '-1' to get a value of -1 for sequential section.
            if(init(trace::currentNodeIndex()-1, op)) {
                if(block1)
                    addBlock(block1);
                if(block2)
                    addBlock(block2);
                if(block3)
                    addBlock(block3);
                timestamp();
            }
        }
        /** Constructor to record a new Task with integers parameters (QR, MGS, BLASGEMM, ...) */
        Task(Operation op, const int *a=NULL, const int *b=NULL, const int *c=NULL, const int *d=NULL, const int *e=NULL):
          buffer_size(0), timeline_(instance()) {
          if(init(trace::currentNodeIndex()-1, op)) {
            if (a) write(*a);
            if (b) write(*b);
            if (c) write(*c);
            if (d) write(*d);
            if (e) write(*e);
            timestamp();
          }
        }
        ~Task();
#else
    public:
        template<typename T> Task(Operation op, HMatrix<T> * block1,
            HMatrix<T> * block2 = NULL, HMatrix<T> * block3 = NULL){}
        Task(Operation op, const int *a=NULL, const int *b=NULL, const int *c=NULL, const int *d=NULL, const int *e=NULL) {}
#endif
    };

#ifdef HMAT_TIMELINE
    /**
     * @brief Set the number of worker.
     * Can be called only once.
     * @param numberOfWorker the number of worker
     * @param rank the process rank (in parallel)
     * @param onlyWorker unless false add a dummy worker for task which are not
     * triggered from a worker.
     */
    void init(int numberOfWorker=1, int rank=0, bool onlyWorker = false);
#else
    void init(int numberOfWorker=1, int rank=0, bool onlyWorker = false){}
#endif
    /** Track pack and unpack */
    void setPackEnabled(bool);
    static Timeline & instance() {
        static Timeline instance;
        return instance;
    }
    void flush();
};

}
