#include "timeline.hpp"
#include "common/my_assert.h"
#include <iomanip>

namespace hmat {

#ifndef __GNUG__
// fwrite is not thread safe in C99 but it is in POSIX and WIN32.
// libstdc++ use a FILE* as backend
#error possibly not thread safe ofstream implementation
#endif

Timeline::Task::~Task() {
    if(workerId_ >= 0) {
        timestamp();
        fwrite(buffer, buffer_size, 1, timeline_.files_[workerId_]);
    }
}

void Timeline::Task::timestamp() {
    Time t = now();
    int64_t tt = t.tv_sec * 1000000000L + t.tv_nsec;
    write(tt);
}

bool Timeline::Task::init(int workerId, Operation op){
    if(timeline_.enabled_ && (timeline_.packEnabled_ || (op != PACK && op != UNPACK))
       && (timeline_.gemmEnabled_ || (op != BLASGEMM))
       && (timeline_.qrEnabled_ || (op != QR && op != MGS))) {
        if(workerId < 0 && !timeline_.onlyWorker_)
            workerId_ = timeline_.files_.size() - 1;
        else
            workerId_=workerId;
        assert(workerId_ >= 0 && workerId_ < timeline_.files_.size());
        write(op);
        return true;
    } else {
        workerId_ = -1;
        return false;
    }
}

void Timeline::Task::addBlock(const IndexSet * rows, const IndexSet * cols) {
    write(rows->offset());
    write(rows->size());
    write(cols->offset());
    write(cols->size());
}

void Timeline::Task::addBlock(int rows, int cols) {
    write(0);
    write(rows);
    write(0);
    write(cols);
}

void Timeline::init(int numberOfWorker, int rank, bool onlyWorker) {
    char * prefix = getenv("HMAT_TIMELINE");
    if(prefix == NULL)
        return;
    if(!onlyWorker_)
        numberOfWorker++;
    assert(numberOfWorker > 0);
    for(int i = 0; i < numberOfWorker; i++) {
        std::ostringstream ss;
        ss << std::setfill('0') << prefix;
        ss << std::setw(2) << rank << "_" << std::setw(2) << i << ".bin";
        FILE * f = fopen(ss.str().c_str(), "wb");
        HMAT_ASSERT(f);
        files_.push_back(f);
    }
    enabled_ = true;

    if (getenv("HMAT_TIMELINE_GEMM")) gemmEnabled_ = true;
    if (getenv("HMAT_TIMELINE_QR")) qrEnabled_ = true;

    Task t(INIT, static_cast<HMatrix<S_t>*>(NULL));
}

void Timeline::setPackEnabled(bool enabled) {
    packEnabled_ = enabled;
}

Timeline::~Timeline() {
    for(int i = 0; i < files_.size(); i++)
        fclose(files_[i]);
}

void Timeline::flush() {
    for(int i = 0; i < files_.size(); i++)
        fflush(files_[i]);
}

}
