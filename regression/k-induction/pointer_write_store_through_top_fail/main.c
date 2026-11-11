// h is rebuilt from a double that holds &tbl[0], and the store through it
// puts &other in tbl. The loop's writes through tbl[i] then
// reach other, so havocking only lists would prove other.off stays below 5.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
unsigned __VERIFIER_nondet_uint(void);

struct reg
{
  int off;
};

struct reg lists[2], other;
struct reg *tbl[2] = {&lists[0], &lists[1]};

int main()
{
  double d = (double)(unsigned long)&tbl[0];
  struct reg **h = (struct reg **)(unsigned long)d;
  *h = &other;
  for (;;)
  {
    unsigned i = __VERIFIER_nondet_uint() & 1u;
    tbl[i]->off = tbl[i]->off + 1;
    __VERIFIER_assert(other.off < 5);
  }
}
