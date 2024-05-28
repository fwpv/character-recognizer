#include "command_interpreter.h"
#include "snn.h"

void RunTests() {
    tests::Propagate();
}

int main(int argc, char** argv) {
    //RunTests();
    ProcessInput(argc, argv);
    return 0;
}