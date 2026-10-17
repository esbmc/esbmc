// d's object reaches a variadic function that reads none of its variadic
// arguments, and a function without a body, neither of which writes through
// it. The index irq loads from it holds no address, and the inductive step
// proves calls < 3.
#include <stdlib.h>
extern void abort(void);
extern unsigned char nondet_uchar(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

struct dev
{
  int n;
  unsigned char buf[8];
};

int calls;

int log_dev(const char *fmt, ...) { return 0; }
void ext(struct dev *);

void irq(struct dev *d, unsigned char data)
{
  unsigned long base = (unsigned long)d;
  log_dev("%p", d);
  ext(d);
  int n = *(int *)base;
  if (n >= 0 && n < 8)
    *(unsigned char *)(base + 4 + n) = data;
  calls = (calls + 1) % 3;
}

int main()
{
  struct dev *d = malloc(sizeof(struct dev));
  d->n = 0;
  for (;;)
  {
    irq(d, nondet_uchar());
    __VERIFIER_assert(calls < 3);
  }
}
