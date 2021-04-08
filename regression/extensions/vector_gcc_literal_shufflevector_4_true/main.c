#include <assert.h>
typedef int v4si __attribute__((__vector_size__(16))); // 4
typedef char v4si2 __attribute__((__vector_size__(32))); // 8

int main() {
    v4si v1 = (v4si){5, 6, 7, 8};
    v4si v2 = (v4si){10, 11, 13, 15};
    v4si expected = (v4si){5,7,10,13};
    v4si r;

    r = __builtin_shufflevector(v1, v2, 0, 2, 4, 6); // 5,7,10,13
    for(int i = 0; i < 4; i++) {
       assert(r[i] == expected[i]);
    }
    return 0;
}