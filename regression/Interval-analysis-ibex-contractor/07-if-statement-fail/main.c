#include <assert.h>

int main() {
    int x = 50;
    float y = 0.1;

    if (x < 1 && y < 1) {
        x = x + y;
        y = y + 1;
    } else if (x >= 20 || x < 0)
        x++;
    else
        x--;
    if(x>0)
        if(x>1)
            if(x>2)
                x++;

    assert(x == y);

    return 0;
}