// Mirrors Solidity's EXACT cpp_new emission pattern (no explicit ctor(base)
// on the heap pointer in main; all state set up via ctor-on-stack-tmp then
// `*new_ptr = tmp` struct copy).  This goto pattern matches 1-to-1 with
// what the Solidity frontend emits for `C base = new C()`.  PASSES — proof
// that the goto-level pattern itself is NOT the bug; the Solidity failure
// must come from something else (e.g. how the contract struct symbol/type
// is registered, value-set init for the global `_ESBMC_Object_C`, etc.).
//
// Build & run:
//   esbmc raw_u256_cpp_sol_pattern.cpp --unwind 3 --no-unwinding-assertions \
//         --no-standard-checks --force-malloc-success --bitwuzla
#include <stdlib.h>
#include <assert.h>

typedef unsigned _ExtInt(256) u256;
extern u256 nondet_u256();
extern unsigned nondet_uint();

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

// Mirror Solidity's struct C — many fields incl anon_pad _ExtInt bit widths.
struct C {
    u256 **grid;
    unsigned _ExtInt(96) pad1;
    unsigned _ExtInt(160) addr;
    unsigned _ExtInt(192) pad3;
    u256 codehash;
    u256 balance;
    u256 code;
    signed char *bind_cname;
    unsigned _ExtInt(192) pad8;
};

static void ctor(struct C *c) {
    c->grid = (u256 **)_alloc_array(3, sizeof(u256 *));
    c->grid[0] = (u256 *)_alloc_array(2, sizeof(u256));
    c->grid[1] = (u256 *)_alloc_array(2, sizeof(u256));
    c->grid[2] = (u256 *)_alloc_array(2, sizeof(u256));
    c->addr = (unsigned _ExtInt(160))nondet_uint();
}

static void set_at(struct C *c, size_t i, size_t j, u256 v) { c->grid[i][j] = v; }
static u256 get_at(struct C *c, size_t i, size_t j) { return c->grid[i][j]; }

static struct C *clone_c(struct C *base) {
    struct C *new_ptr = new struct C;   // cpp_new (no init — skip default ctor)
    struct C tmp;
    ctor(&tmp);
    *new_ptr = tmp;                     // direct struct ASSIGN (no operator=)
    struct C *c = new_ptr;
    *c = *base;                         // direct struct ASSIGN
    c->addr = (unsigned _ExtInt(160))nondet_uint();
    __ESBMC_assume(c->addr != base->addr);
    c->grid = (u256 **)_alloc_array(3, sizeof(u256 *));
    c->grid[0] = (u256 *)_arrcpy(base->grid[0], 2, sizeof(u256));
    c->grid[1] = (u256 *)_arrcpy(base->grid[1], 2, sizeof(u256));
    c->grid[2] = (u256 *)_arrcpy(base->grid[2], 2, sizeof(u256));
    return c;
}

int main() {
    u256 a = nondet_u256();
    __ESBMC_assume(a != 0);

    // Solidity's exact emission for `C base = new C()`:
    //   - cpp_new allocates heap via MALLOC
    //   - stack tmp NONDET-initialised then ctor(&tmp) fills grid
    //   - *new_ptr = tmp  (direct struct ASSIGN, copies grid pointer)
    //   - base = new_ptr
    //   - (NO explicit ctor(base) on the heap pointer)
    struct C *new_ptr = new struct C;
    struct C tmp;
    ctor(&tmp);
    *new_ptr = tmp;
    struct C *base = new_ptr;

    set_at(base, 0, 0, a);
    assert(get_at(base, 0, 0) == a);   // base round-trip passes
    struct C *cl = clone_c(base);
    assert(get_at(cl, 0, 0) == a);      // clone round-trip also passes here
    return 0;
}
