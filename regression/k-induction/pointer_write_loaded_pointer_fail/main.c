// The write through t[i] reaches lists, not tbl. Naming no object for t[i]
// would leave lists unhavocked and the inductive step would prove the bound.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
unsigned __VERIFIER_nondet_uint(void);

struct reg
{
  int off;
};

struct reg lists[2];
struct reg *tbl[2] = {&lists[0], &lists[1]};

int main()
{
  struct reg **t = tbl;
  for (;;)
  {
    unsigned i = __VERIFIER_nondet_uint() & 1u;
    t[i]->off = t[i]->off + 1;
    __VERIFIER_assert(lists[i].off < 10);
  }
}
