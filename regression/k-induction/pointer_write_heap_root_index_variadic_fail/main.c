// set reads d and B back with va_arg and stores B's address in d's object, so
// what the loop loads from it reaches B. Unless the inductive step havocs B's
// object, B[1] keeps its pre-loop 0 and the step proves B[1] < 3.
#include <stdarg.h>
#include <stdlib.h>
#include <assert.h>
extern int nondet_int(void);

struct dev
{
  long base;
};

void set(int k, ...)
{
  va_list ap;
  va_start(ap, k);
  struct dev *x = va_arg(ap, struct dev *);
  int *y = va_arg(ap, int *);
  va_end(ap);
  x->base = (long)y;
}

int main()
{
  struct dev *d = malloc(sizeof(struct dev));
  int *B = malloc(2 * sizeof(int));
  B[0] = 0;
  B[1] = 0;
  set(2, d, B);
  while (nondet_int())
  {
    long base = *(long *)(unsigned long)d;
    int *r = (int *)(base + 4);
    *r = *r + 1;
    assert(B[1] < 3);
  }
  return 0;
}
