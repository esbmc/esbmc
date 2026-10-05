// As pointer_write_heap_root_reassigned, with the loop shaped as CIL emits it:
// the first iteration is entered by a jump to the test at the bottom.
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
  int i = 0;
  unsigned long base;
  unsigned long reg;
  goto test;
body:
  base = (unsigned long)d;
  reg = base + 4UL * i;
  *(int *)reg = i;
  i++;
test:
  if (i < 2)
    goto body;
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
