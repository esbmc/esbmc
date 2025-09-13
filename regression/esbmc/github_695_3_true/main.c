#include <assert.h>

/*
clang -fsanitize=undefined,integer,nullability -O2 main.c

UndefinedBehaviorSanitizer:DEADLYSIGNAL
==14107==ERROR: UndefinedBehaviorSanitizer: SEGV on unknown address 0x0000bddcd280 (pc 0x56533e3c1d98 bp 0x0000bddcd280 sp 0x7ffebddcd280 T14107)
==14107==The signal is caused by a WRITE memory access.
    #0 0x56533e3c1d98 in main (/home/lucas/ESBMC_Project/esbmc/regression/esbmc/github_695_3_true/a.out+0x2cd98) (BuildId: 5573efe4f03e606b463871256f64cb90b4309ff7)
    #1 0x7facfcf8ad8f in __libc_start_call_main csu/../sysdeps/nptl/libc_start_call_main.h:58:16
    #2 0x7facfcf8ae3f in __libc_start_main csu/../csu/libc-start.c:392:3
    #3 0x56533e399374 in _start (/home/lucas/ESBMC_Project/esbmc/regression/esbmc/github_695_3_true/a.out+0x4374) (BuildId: 5573efe4f03e606b463871256f64cb90b4309ff7)

UndefinedBehaviorSanitizer can not provide additional info.
SUMMARY: UndefinedBehaviorSanitizer: SEGV (/home/lucas/ESBMC_Project/esbmc/regression/esbmc/github_695_3_true/a.out+0x2cd98) (BuildId: 5573efe4f03e606b463871256f64cb90b4309ff7) in main
==14107==ABORTING

*/
struct obj {
    int a;
    int b;
};

#define A_ELEMENT(o) o
#define B_ELEMENT(o) o + sizeof(int)

int main() {
    struct obj A[4];
    unsigned int obj_ptr = (unsigned int)&A[0];

    for (int i = 0; i < 4; i++)
    {
        unsigned int var = obj_ptr + i*sizeof(struct obj);
        *((int*)(A_ELEMENT(var))) = i;
        *((int*)(B_ELEMENT(var))) = i+1;

        struct obj *ref = (struct obj*) var;
        assert(ref->a == i);
        assert(ref->b == i+1);
    }

    for (int i = 0; i < 4; i++)
    {
        assert(A[i].a == i);
        assert(A[i].b == i+1);
    }


    return 0;
}
