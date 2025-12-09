#include <assert.h>

int main() {
    int x = 30;
    float y = 0.1;

    assert(x == 30 && y >= 0.1 && x+y >= 30.1 );

    return 0;
}