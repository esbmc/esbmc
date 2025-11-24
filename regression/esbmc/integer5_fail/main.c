#include <limits.h>

int main() {
    int a = INT_MIN;
    int b = -1;
    
    // This operation results in signed integer overflow
    int result = a / b;

    return 0;
}

