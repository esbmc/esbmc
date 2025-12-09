#include <assert.h>

int main() {
    int x = 50;
    float y = 0.1;

    if (x <= 1) {
        x = x + y;
        y = y + 1;
    } else if (x >= 20)
        x++;

    while (x > 40)
        x--;

    while (y < 1)
        y++;

    assert(x != y);

    return 0;
}