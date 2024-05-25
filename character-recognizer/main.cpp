#include "neural_matrix.h"
#include "profiler.h"
#include "training.h"

void RunTests() {
    {
        LOG_DURATION("Propagate");
        tests::Propagate();
    }
}

int main(/*int argc, char** argv*/) {
    RunTests();
    RunExperiment1();

    return 0;
}