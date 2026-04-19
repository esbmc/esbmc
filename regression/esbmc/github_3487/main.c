#include <assert.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint8_t mem[2];
    uint16_t* ptr;
} box_t;

int main(void) {
    box_t b = {0};
    uint16_t val = 42;
    b.ptr = &val;

    int16_t v = 1;
    memcpy(b.mem, &v, 2);

    assert(*b.ptr == 42);
    return 0;
}
