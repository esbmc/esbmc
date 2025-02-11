#include <assert.h>

int main() {
    int x = 50;
    float y = 0.1;

    if (x < 1 && y < 1) {
        x = x + y;
        y = y + 1;
    } else if (x >= 20 || x < 0)
        x++;

    while (x > 40 && y > 1)
        x--;

    while (y < 1 || x < 3)
        y++;

    assert(x != y && x != 0 || y == 0);

    return 0;
}