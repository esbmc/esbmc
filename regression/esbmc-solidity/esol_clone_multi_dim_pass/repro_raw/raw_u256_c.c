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

struct C {
    u256 **grid;
    unsigned long _ESBMC_addr;
};

static void ctor(struct C *c) {
    c->grid = (u256 **)_alloc_array(3, sizeof(u256 *));
    c->grid[0] = (u256 *)_alloc_array(2, sizeof(u256));
    c->grid[1] = (u256 *)_alloc_array(2, sizeof(u256));
    c->grid[2] = (u256 *)_alloc_array(2, sizeof(u256));
    c->_ESBMC_addr = 0x42;
}

static void set_at(struct C *c, size_t i, size_t j, u256 v) { c->grid[i][j] = v; }
static u256 get_at(struct C *c, size_t i, size_t j) { return c->grid[i][j]; }

static struct C *clone_c(struct C *base) {
    struct C *new_ptr = malloc(sizeof(struct C));
    struct C tmp;
    ctor(&tmp);
    *new_ptr = tmp;          // direct struct assign
    struct C *c = new_ptr;
    *c = *base;              // direct struct assign
    c->_ESBMC_addr = nondet_ul();
    __ESBMC_assume(c->_ESBMC_addr != base->_ESBMC_addr);
    c->grid = (u256 **)_alloc_array(3, sizeof(u256 *));
    c->grid[0] = (u256 *)_arrcpy(base->grid[0], 2, sizeof(u256));
    c->grid[1] = (u256 *)_arrcpy(base->grid[1], 2, sizeof(u256));
    c->grid[2] = (u256 *)_arrcpy(base->grid[2], 2, sizeof(u256));
    return c;
}

int main() {
    u256 a = nondet_u256();
    __ESBMC_assume(a != 0);

    struct C *base = malloc(sizeof(struct C));
    ctor(base);
    set_at(base, 0, 0, a);
    assert(get_at(base, 0, 0) == a);
    struct C *cl = clone_c(base);
    assert(get_at(cl, 0, 0) == a);
    return 0;
}
