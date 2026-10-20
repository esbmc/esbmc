// step's loop reassigns p before writing through it, but the write after the
// loop reads p too: the inductive step may leave the loop right after its
// havoc of p, so p cannot be traced to its assignments in the body.
#include <stdlib.h>
#include <assert.h>
extern int nondet_int(void);
int *d, *e;
unsigned n;
void step(void)
{
  int *p = d;
  int j = nondet_int();
  while (j > 0)
  {
    p = e;
    *p = 1;
    j = 0;
  }
  if (n == 3)
    *p = 10;
}
int main()
{
  d = malloc(sizeof(int));
  e = malloc(sizeof(int));
  *d = 0;
  while (n < 6)
  {
    *d = 0;
    step();
    assert(*d != 10);
    n++;
  }
}
