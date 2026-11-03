// d's object holds B's address as an integer, so what the loop loads from it
// is no offset but an address into B. Taking it for one would havoc nothing
// for the write and prove B[1] < 3.
#include <stdlib.h>
#include <assert.h>
extern int nondet_int(void);

struct dev
{
  long base;
};

int main()
{
  struct dev *d = malloc(sizeof(struct dev));
  int *B = malloc(2 * sizeof(int));
  B[0] = 0;
  B[1] = 0;
  d->base = (long)B;
  while (nondet_int())
  {
    long base = *(long *)(unsigned long)d;
    int *r = (int *)(base + 4);
    *r = *r + 1;
    assert(B[1] < 3);
  }
  return 0;
}
