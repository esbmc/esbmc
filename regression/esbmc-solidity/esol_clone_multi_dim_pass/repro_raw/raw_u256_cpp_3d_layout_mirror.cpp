// Raw C++ 3D that mirrors Solidity's contract struct layout PRECISELY,
// including all anon_pad fields, $address/$balance/$codehash/$code/$bind_cname,
// plus global msg_sender, _ESBMC_Object_C, and the harness pattern
// (base$bind = C, msg_sender = this->$address).
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef unsigned _ExtInt(256) u256;
typedef unsigned _ExtInt(160) addr_t;
extern u256 nondet_u256();
extern unsigned long nondet_ul();
extern unsigned int nondet_uint();

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

static void *_arrcpy_2d(void *from, size_t outer, size_t inner, size_t sz) {
__ESBMC_HIDE:;
    void *dst_outer_raw = _alloc_array(outer, sizeof(void *));
    __builtin_memcpy(dst_outer_raw, from, outer * sizeof(void *));
    void **dst = (void **)dst_outer_raw;
    for (size_t i = 0; i < outer; i++)
        dst[i] = _arrcpy(dst[i], inner, sz);
    return dst_outer_raw;
}

// Mirror Solidity's contract struct layout byte-for-byte.
struct C {
    u256 ***arr;                              // 8 bytes, pointer
    unsigned _ExtInt(96) pad_1;          // 12 bytes
    addr_t $address;                          // 20 bytes
    unsigned _ExtInt(192) pad_3;         // 24 bytes
    u256 $codehash;                           // 32 bytes
    u256 $balance;                            // 32 bytes
    u256 $code;                               // 32 bytes
    signed char *_ESBMC_bind_cname;           // 8 bytes
    unsigned _ExtInt(192) pad_8;         // 24 bytes
};

// Solidity-style globals
static addr_t msg_sender;
static signed char global_C_tag[] = "C";

static void ctor(struct C *c) {
    c->arr = (u256 ***)_alloc_array(3, sizeof(u256 **));
    c->arr[0] = (u256 **)_alloc_array(2, sizeof(u256 *));
    c->arr[0][0] = (u256 *)_alloc_array(2, sizeof(u256));
    c->arr[0][1] = (u256 *)_alloc_array(2, sizeof(u256));
    c->arr[1] = (u256 **)_alloc_array(2, sizeof(u256 *));
    c->arr[1][0] = (u256 *)_alloc_array(2, sizeof(u256));
    c->arr[1][1] = (u256 *)_alloc_array(2, sizeof(u256));
    c->arr[2] = (u256 **)_alloc_array(2, sizeof(u256 *));
    c->arr[2][0] = (u256 *)_alloc_array(2, sizeof(u256));
    c->arr[2][1] = (u256 *)_alloc_array(2, sizeof(u256));
    c->$address = (addr_t)(0x42);
}

static void setAt(struct C *c, size_t i, size_t j, size_t k, u256 v) {
    c->arr[i][j][k] = v;
}
static u256 get(struct C *c, size_t i, size_t j, size_t k) {
    return c->arr[i][j][k];
}

static struct C *_ESBMC_clone_C(struct C *base) {
__ESBMC_HIDE:;
    struct C *c;
    struct C *new_ptr$1;
    new_ptr$1 = new struct C;
    struct C tmp$2;
    ctor(&tmp$2);
    *new_ptr$1 = tmp$2;
    c = new_ptr$1;
    *c = *base;
    c->$address = (addr_t)nondet_uint();
    __ESBMC_assume(c->$address != base->$address);

    c->arr = (u256 ***)_alloc_array(3, sizeof(u256 **));
    c->arr[0] = (u256 **)_arrcpy_2d(base->arr[0], 2, 2, sizeof(u256));
    c->arr[1] = (u256 **)_arrcpy_2d(base->arr[1], 2, 2, sizeof(u256));
    c->arr[2] = (u256 **)_arrcpy_2d(base->arr[2], 2, 2, sizeof(u256));
    return c;
}

int main() {
    u256 a = nondet_u256();
    __ESBMC_assume(a != 0);

    // Mirror the check() body from Solidity
    struct C *base;
    struct C *new_ptr$1;
    new_ptr$1 = new struct C;
    struct C tmp$2;
    ctor(&tmp$2);
    *new_ptr$1 = tmp$2;
    base = new_ptr$1;
    base->_ESBMC_bind_cname = global_C_tag;

    addr_t old_sender = msg_sender;
    msg_sender = base->$address;         // route calls through
    setAt(base, 0, 0, 0, a);
    msg_sender = old_sender;

    struct C *clone = _ESBMC_clone_C(base);

    old_sender = msg_sender;
    msg_sender = clone->$address;
    u256 got = get(clone, 0, 0, 0);
    assert(got == a);
    msg_sender = old_sender;
    return 0;
}
