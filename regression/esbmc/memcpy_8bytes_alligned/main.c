#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

int main() {
    // 16-byte aligned buffers (malloc guarantees this)
    uint64_t *src = malloc(32);
    uint64_t *dst = malloc(32);
    
    if (!src || !dst) {
      if (src) free(src);
      if (dst) free(dst);
        return 1;
    }

    src[0] = 0x0123456789ABCDEF;
    src[1] = 0xFEDCBA9876543210;
    
    memcpy(dst, src, 16);  // Should use 8-byte copies
    
    assert(dst[0] == src[0] && dst[1] == src[1]);
    free(src); free(dst);
    return 0;
}
