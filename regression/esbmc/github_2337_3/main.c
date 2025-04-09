#include <assert.h>
#include <stdbool.h>

int main() {
    bool x = true;   // Equivalent to 1
    int result = -x; // Should be -1
    assert(result == -1);
    return 0;
}

