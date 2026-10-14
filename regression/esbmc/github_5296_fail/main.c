// #5296 companion: the signature filter must keep the *correct* target.
// Same over-approximated heap function pointer as github_5296, but the
// legitimately-dispatched 1-argument uhash asserts its argument is non-null and
// the key may be null. The assertion is reachable only if uhash is still a call
// target after filtering, so this FAILED result guards against the filter
// over-pruning and silently skipping the real call.
#include <stdint.h>
#include <stdlib.h>
extern void abort(void);
extern unsigned long __VERIFIER_nondet_ulong(void);

uint64_t uhash(const void *a) // 1-argument
{
  __ESBMC_assert(a != NULL, "uhash argument must be non-null");
  return (uint64_t)a ? (uint64_t)a : 1ull;
}

_Bool eqnn(const void *a, const void *b) // 2-argument
{
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
  void *k = (void *)(uintptr_t)__VERIFIER_nondet_ulong(); // may be null
  uint64_t h = do_hash(&t, k);
  return 0;
}
