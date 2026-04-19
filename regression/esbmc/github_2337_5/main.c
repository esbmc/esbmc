#include <stdbool.h>
#include <limits.h>

int main() {
    bool x = true;
    int max_val = INT_MAX;
    int result = -(max_val + x); // Should detect overflow
    return 0;
}

