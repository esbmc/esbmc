// #5296: an indirect call through a function pointer read from a
// symbolic-size heap object gets an over-approximated value-set that lists
// address-taken functions of unrelated arity. Dispatching the 1-argument hash
// call to the 2-argument equals function silently models the missing argument
// as nondet ("missing argument for parameter ...; modelled as nondet").
//
// The pointer provably holds uhash, so the program is SUCCESSFUL either way;
// the bug is the spurious wrong-arity candidate. This test passes only when
// that candidate is filtered out (no missing-argument warning).
#include <stdint.h>
#include <stdlib.h>
extern void abort(void);
extern unsigned long __VERIFIER_nondet_ulong(void);

uint64_t uhash(const void *a) // 1-argument
{
  return (uint64_t)a ? (uint64_t)a : 1ull;
}

_Bool eqnn(const void *a, const void *b) // 2-argument
{
  __ESBMC_assert(b != NULL, "b must not be a dropped (nondet) argument");
  return a == b;
}

typedef uint64_t (*hfn)(const void *);
typedef _Bool (*efn)(const void *, const void *);
struct state
{
  hfn hf;
  efn ef;
  size_t n;
};
struct table
{
  struct state *p;
};

static void alloc_table(struct table *t)
{
  size_t n = __VERIFIER_nondet_ulong();
  if (!(n && n <= 4))
    abort();
  struct state *s = malloc(sizeof(struct state) + n * 16); // symbolic-size heap
  if (!s)
    abort();
  s->hf = uhash;
  s->ef = eqnn;
  s->n = n;
  t->p = s;
}

static uint64_t do_hash(struct table *t, const void *k)
{
  return t->p->hf(k); // 1-argument indirect call through a heap-read pointer
}

int main(void)
{
  struct table t;
  alloc_table(&t);
  void *k = (void *)(uintptr_t)__VERIFIER_nondet_ulong();
  if (!k)
    abort();
  uint64_t h = do_hash(&t, k);
  __ESBMC_assert(h != 0, "uhash never returns 0");
  return 0;
}
