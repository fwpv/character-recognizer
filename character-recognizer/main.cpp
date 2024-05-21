#include "neural_matrix.h"
#include "profiler.h"

#include <iostream>
#include <iomanip>
#include <vector>

void RunTests() {
    {
        LOG_DURATION("Propagate");
        tests::Propagate();
    }
}

int main(/*int argc, char** argv*/) {
    RunTests();

    return 0;
}