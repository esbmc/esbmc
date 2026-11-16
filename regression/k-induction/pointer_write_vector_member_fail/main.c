// h is rebuilt from a vector lane inside a constant struct, and the store
// through it puts &other in tbl. Havocking only lists would prove other.off
// stays below 5.
extern void abort(void);
void reach_error()
{
}
void __VERIFIER_assert(int c)
{
  if (!c)
  {
  ERROR:
    {
      reach_error();
      abort();
    }
  }
}
unsigned __VERIFIER_nondet_uint(void);

typedef unsigned long v2 __attribute__((vector_size(16)));

struct reg
{
  int off;
};

struct reg lists[2], other;
struct reg *tbl[2] = {&lists[0], &lists[1]};

int main()
{
  struct
  {
    v2 a;
    int k;
  } s = {{(unsigned long)&tbl[0], 0}, 0};
  struct reg **h = (struct reg **)s.a[0];
  *h = &other;
  for (;;)
  {
    unsigned i = __VERIFIER_nondet_uint() & 1u;
    tbl[i]->off = tbl[i]->off + 1;
    __VERIFIER_assert(other.off < 5);
  }
}
