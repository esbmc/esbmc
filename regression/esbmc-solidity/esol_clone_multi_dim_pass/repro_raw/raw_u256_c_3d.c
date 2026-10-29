// 3D extension of raw_u256_c.c.
// Solidity: uint256[2][2][3] arr inside contract C.
// Tests whether the value-set loss reproduces in pure C when the
// (A) fresh-outer-alloc + (B) per-slot write + (C) later index-read
// pattern is applied at the outermost layer, mirroring the walker's
// emission for a 3D field.
//
// If this program fails → backend bug reproducible without Solidity frontend.
// If this program passes → break is specific to Solidity-frontend attributes.
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef unsigned _ExtInt(256) u256;
extern u256 nondet_u256();
extern unsigned long nondet_ul();

static void *_alloc_array(size_t count, size_t elem_size) {
__ESBMC_HIDE:;
    size_t total = sizeof(size_t) + count * elem_size;
    size_t *block = (size_t *)calloc(1, total);
    block[0] = count;
    return (void *)(block + 1);
}

static void *_arrcpy(void *from, size_t n, size_t sz) {
__ESBMC_HIDE:;
    void *to = _alloc_array(n, sz);
    u256 *s = (u256 *)from;
    u256 *d = (u256 *)to;
    for (size_t i = 0; i < n; i++) d[i] = s[i];
    return to;
}

// 3D pointer-of-pointer-of-pointer: u256[2][2][3] → u256 ***
struct C {
    u256 ***arr;
    unsigned long _ESBMC_addr;
};

static void ctor(struct C *c) {
    c->arr = (u256 ***)_alloc_array(3, sizeof(u256 **));
    for (size_t i = 0; i < 3; i++) {
        c->arr[i] = (u256 **)_alloc_array(2, sizeof(u256 *));
        for (size_t j = 0; j < 2; j++) {
            c->arr[i][j] = (u256 *)_alloc_array(2, sizeof(u256));
        }
    }
    c->_ESBMC_addr = 0x42;
}

static void set_at(struct C *c, size_t i, size_t j, size_t k, u256 v) {
    c->arr[i][j][k] = v;
}
static u256 get_at(struct C *c, size_t i, size_t j, size_t k) {
    return c->arr[i][j][k];
}

// Mirrors the 3D walker emission:
// (A) c->arr = alloc_array(3, 8)            fresh outer
// (B) c->arr[i] = <some heap pointer>        per-slot write, for i=0..2
// (C) later: c->arr[i][j][k] read            should equal what was written
static struct C *clone_c(struct C *base) {
    struct C *new_ptr = malloc(sizeof(struct C));
    struct C tmp;
    ctor(&tmp);
    *new_ptr = tmp;
    struct C *c = new_ptr;
    *c = *base;
    c->_ESBMC_addr = nondet_ul();
    __ESBMC_assume(c->_ESBMC_addr != base->_ESBMC_addr);

    // (A) fresh outer allocation
    c->arr = (u256 ***)_alloc_array(3, sizeof(u256 **));

    // (B) per-slot writes — for each i, freshly allocate an inner 2x2 block
    //     and copy from base
    for (size_t i = 0; i < 3; i++) {
        u256 **inner = (u256 **)_alloc_array(2, sizeof(u256 *));
        for (size_t j = 0; j < 2; j++) {
            inner[j] = (u256 *)_arrcpy(base->arr[i][j], 2, sizeof(u256));
        }
        c->arr[i] = inner;
    }
    return c;
}

int main() {
    u256 a = nondet_u256();
    __ESBMC_assume(a != 0);

    struct C *base = malloc(sizeof(struct C));
    ctor(base);
    set_at(base, 0, 0, 0, a);
    assert(get_at(base, 0, 0, 0) == a);
    struct C *cl = clone_c(base);
    // (C) later read — if value-set is lost, this returns nondet.
    assert(get_at(cl, 0, 0, 0) == a);
    return 0;
}
