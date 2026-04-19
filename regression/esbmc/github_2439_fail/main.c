#include <math.h>
#include <assert.h>

int main() {
    assert(acos(1.0) >= 0);
    assert(acos(0.0) < 1.57);
    return 0;
}