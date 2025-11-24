#include <assert.h>
#include <stdbool.h>

int main() {
    bool x = false;  // Equivalent to 0
    int result = -x; // Should be 0
    assert(!x);
    return 0;
}

