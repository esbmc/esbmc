#include <assert.h>
#include <stdint.h>

int main() {
    uint32_t x = 0;
    uint32_t y = 2147483648U;
    
    assert(((x * 2U) == (y * 2U)) == (x == y));
    
    return 0;
}
