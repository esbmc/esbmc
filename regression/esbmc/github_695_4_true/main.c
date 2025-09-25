#include <assert.h>

/*
$ clang -fsanitize=undefined,integer,nullability -O2 main.c

==14411==ERROR: UndefinedBehaviorSanitizer: SEGV on unknown address 0x0000d7262a60 (pc 0x564b1cbd7db3 bp 0x000000000000 sp 0x7fffd7262a60 T14411)
==14411==The signal is caused by a WRITE memory access.
    #0 0x564b1cbd7db3 in main (/home/lucas/ESBMC_Project/esbmc/regression/esbmc/github_695_4_true/a.out+0x2cdb3) (BuildId: b0b6e29764cf6f5ca8d9fcff624b7a7ea8b5b9d1)
    #1 0x7fdd2f6b9d8f in __libc_start_call_main csu/../sysdeps/nptl/libc_start_call_main.h:58:16
    #2 0x7fdd2f6b9e3f in __libc_start_main csu/../csu/libc-start.c:392:3
    #3 0x564b1cbaf374 in _start (/home/lucas/ESBMC_Project/esbmc/regression/esbmc/github_695_4_true/a.out+0x4374) (BuildId: b0b6e29764cf6f5ca8d9fcff624b7a7ea8b5b9d1)

UndefinedBehaviorSanitizer can not provide additional info.
SUMMARY: UndefinedBehaviorSanitizer: SEGV (/home/lucas/ESBMC_Project/esbmc/regression/esbmc/github_695_4_true/a.out+0x2cdb3) (BuildId: b0b6e29764cf6f5ca8d9fcff624b7a7ea8b5b9d1) in main
==14411==ABORTING
*/

union foo {
    int elem1;
    long long elem2;
};

struct obj {
    union foo a; // this will use elem1
    union foo b; // this will use elem2
};

#define A_ELEMENT(o) o
#define B_ELEMENT(o) o + sizeof(union foo)

int main() {
    struct obj A[4];
    unsigned int obj_ptr = (unsigned int)&A[0];

    for (int i = 0; i < 4; i++)
    {
        unsigned int var = obj_ptr + i*sizeof(struct obj);
        ((union foo*)(A_ELEMENT(var)))->elem1 = i;
        ((union foo*)(B_ELEMENT(var)))->elem2 = i+1;

        struct obj *ref = (struct obj*) var;
        assert(ref->a.elem1 == i);
        assert(ref->b.elem2 == i+1);
    }

    for (int i = 0; i < 4; i++)
    {
        assert(A[i].a.elem1 == i);
        assert(A[i].b.elem2 == i+1);
    }


    return 0;
}
