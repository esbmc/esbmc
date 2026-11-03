// Inlining program_offset turns t[idx] into a load through &tbl[0] + idx.
// Havocking lists, what tbl's elements point to, proves the bound. stdint.h
// links the pthread models, whose unreachable stores must not widen tbl.
#include <stdint.h>
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
uint32_t __VERIFIER_nondet_uint(void);

struct reg
{
  uint32_t off;
};

struct reg lists[4];
struct reg *tbl[4] = {&lists[0], &lists[1], &lists[2], &lists[3]};

static void program_offset(struct reg **t, uint32_t idx, uint32_t base)
{
  t[idx]->off = base % 10;
}

int main()
{
  for (;;)
  {
    uint32_t i = __VERIFIER_nondet_uint();
    program_offset(tbl, i & 3u, i);
    __VERIFIER_assert(lists[i & 3u].off < 10);
  }
}
