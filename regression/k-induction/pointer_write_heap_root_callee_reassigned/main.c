// probe's loop reassigns base and reg before every read, so the havoc its
// inductive step gives them is never read and main's loop traces the write
// through them to d. The step havocs d's object whole and proves calls < 3.
#include <stdlib.h>
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

struct dev
{
  int regs[2];
};

struct dev *d;
int calls;

void probe(void)
{
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
  d = malloc(sizeof(struct dev));
  for (;;)
  {
    probe();
    __VERIFIER_assert(calls < 3);
  }
}
