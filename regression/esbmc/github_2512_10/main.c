#include <stdlib.h>
#include <stddef.h>

// Different alignment requirements
struct __attribute__((aligned(16))) aligned_16 {
    char data;
};

struct __attribute__((aligned(32))) aligned_32 {
    int value;
};

struct __attribute__((packed, aligned(8))) packed_aligned {
    char a;
    int b;
    short c;
};

// Mixed alignment and packing
struct complex_alignment {
    char prefix;
    struct __attribute__((aligned(16))) aligned_16 a16;
    struct __attribute__((packed)) {
        char x;
        int y;
        char z;
    } packed_inner;
    struct __attribute__((aligned(32))) aligned_32 a32;
    char suffix;
};

// Bitfield alignment edge cases
struct bitfield_complex {
    int header;
    unsigned int bf1 : 3;
    unsigned int bf2 : 5;
    unsigned int bf3 : 24;  // Spans multiple bytes
    char middle;
    unsigned int bf4 : 1;
    unsigned int bf5 : 7;
    unsigned int bf6 : 8;
    int trailer;
};

// Architecture-specific alignment (simulate different targets)
struct arch_specific {
    char c1;
    void *ptr;      // Different sizes on 32/64 bit
    long l;         // Different sizes on different archs
    double d;       // Usually 8-byte aligned
    char c2;
};

int main() {
    // Test complex alignment recovery
    struct complex_alignment *ca = malloc(sizeof *ca);
    ca->prefix = 'P';
    ca->a16.data = 'A';
    ca->packed_inner.x = 'X';
    ca->packed_inner.y = 0x12345678;
    ca->a32.value = 0xDEADBEEF;
    ca->suffix = 'S';
    
    // Container_of with aligned members
    struct aligned_16 *a16_ptr = &ca->a16;
    void *tmp1 = ((void*)a16_ptr) - offsetof(struct complex_alignment, a16);
    struct complex_alignment *recovered1 = (struct complex_alignment*)tmp1;
    
    struct aligned_32 *a32_ptr = &ca->a32;
    void *tmp2 = ((void*)a32_ptr) - offsetof(struct complex_alignment, a32);
    struct complex_alignment *recovered2 = (struct complex_alignment*)tmp2;
    
    // Test bitfield structure
    struct bitfield_complex *bf = malloc(sizeof *bf);
    bf->header = 0x1234;
    bf->bf1 = 7;
    bf->bf2 = 31;
    bf->bf3 = 0xFFFFFF;
    bf->middle = 'M';
    bf->trailer = 0x5678;
    
    // Container_of with bitfield-containing struct
    char *middle_ptr = &bf->middle;
    void *tmp3 = ((void*)middle_ptr) - offsetof(struct bitfield_complex, middle);
    struct bitfield_complex *bf_recovered = (struct bitfield_complex*)tmp3;
    
    // Verify values
    char prefix = recovered1->prefix;
    char suffix = recovered2->suffix;
    int header = bf_recovered->header;
    int trailer = bf_recovered->trailer;
    
    free(ca);
    free(bf);
    return 0;
}

