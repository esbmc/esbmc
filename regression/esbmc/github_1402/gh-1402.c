#include <assert.h>

struct Granule {
    int value;
};

int main() {
    struct Granule granules[10] = {
        {10}, {20}, {30}, {40}, {50}, {60}, {70}, {80}, {90}, {100}
    };

    unsigned long int offset = (&granules[0] + 6) - &granules[0];
    int n = offset;
    assert(n == 6);

    return 0;
}
