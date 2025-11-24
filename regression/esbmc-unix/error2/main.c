#define _GNU_SOURCE
#include <error.h>
#include <assert.h>

int main(void) {
    // This will terminate the program with exit code 1
    error(1, 0, "This is a fatal error");

    // This assertion is never reached
    assert(0 && "Should not reach here");

    return 0;
}

