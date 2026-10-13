// probe's loop writes through integers it computes from d in every
// iteration before using them, and d changes only between calls of probe.
// The inductive step havocs d's object whole and proves calls < 3.
#include <stdlib.h>
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

struct dev
{
  int regs[2];
};

int calls;

void *alloc(unsigned n) { return malloc(n); }

void probe(void)
{
  struct dev *d = alloc(sizeof(struct dev));
  for (int i = 0; i < 2; ++i)
  {
    unsigned long base = (unsigned long)d;
    unsigned long reg = base + 4UL * i;
    *(int *)reg = i;
  }
  calls = (calls + 1) % 3;
}

int main()
{
  for (;;)
  {
    probe();
    __VERIFIER_assert(calls < 3);
  }
}
