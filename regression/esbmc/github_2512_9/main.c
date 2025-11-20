#include <stdlib.h>
#include <stddef.h>

struct level7 {
    int data7;
    char marker7;
};

struct level6 {
    double val6;
    struct level7 l7;
    int end6;
};

struct level5 {
    struct level6 l6;
    short s5;
};

struct level4 {
    char prefix4;
    struct level5 l5;
    long suffix4;
};

struct level3 {
    struct level4 l4;
    float f3;
};

struct level2 {
    int header2;
    struct level3 l3;
    char tail2[8];
};

struct level1 {
    struct level2 l2;
    void *ptr1;
};

struct root {
    char magic[4];
    struct level1 l1;
    int checksum;
};

int main() {
    struct root *r = malloc(sizeof *r);
    
    // Initialize deeply nested structure
    r->magic[0] = 'T'; r->magic[1] = 'E'; r->magic[2] = 'S'; r->magic[3] = 'T';
    r->l1.l2.header2 = 0x1234;
    r->l1.l2.l3.l4.prefix4 = 'P';
    r->l1.l2.l3.l4.l5.l6.val6 = 3.14159;
    r->l1.l2.l3.l4.l5.l6.l7.data7 = 42;
    r->l1.l2.l3.l4.l5.l6.l7.marker7 = 'M';
    r->checksum = 0xDEADBEEF;
    
    // Get pointer to deeply nested member
    struct level7 *deep_ptr = &r->l1.l2.l3.l4.l5.l6.l7;
    
    // Recover through multiple container_of operations
    void *tmp6 = ((void*)deep_ptr) - offsetof(struct level6, l7);
    struct level6 *l6_recovered = (struct level6*)tmp6;
    
    void *tmp5 = ((void*)l6_recovered) - offsetof(struct level5, l6);
    struct level5 *l5_recovered = (struct level5*)tmp5;
    
    void *tmp4 = ((void*)l5_recovered) - offsetof(struct level4, l5);
    struct level4 *l4_recovered = (struct level4*)tmp4;
    
    void *tmp3 = ((void*)l4_recovered) - offsetof(struct level3, l4);
    struct level3 *l3_recovered = (struct level3*)tmp3;
    
    void *tmp2 = ((void*)l3_recovered) - offsetof(struct level2, l3);
    struct level2 *l2_recovered = (struct level2*)tmp2;
    
    void *tmp1 = ((void*)l2_recovered) - offsetof(struct level1, l2);
    struct level1 *l1_recovered = (struct level1*)tmp1;
    
    void *tmp_root = ((void*)l1_recovered) - offsetof(struct root, l1);
    struct root *root_recovered = (struct root*)tmp_root;
    
    // Verify recovery worked
    char magic0 = root_recovered->magic[0];
    int checksum = root_recovered->checksum;
    int data7 = root_recovered->l1.l2.l3.l4.l5.l6.l7.data7;
    
    free(r);
    return 0;
}
