// The loop allocates only on its first iteration and keeps the object in p.
// The inductive step havocs p into a pointer symex cannot resolve, so the
// object this loop allocates is still no object the step may skip: every
// write through p would be dropped and the step would prove *p < 3.
#include <stdlib.h>
#include <assert.h>
extern int nondet_int(void);
int main()
{
  int *p = 0;
  while (nondet_int())
  {
    if (!p)
    {
      p = malloc(sizeof(int));
      *p = 0;
    }
    *p = *p + 1;
    assert(*p < 3);
  }
  return 0;
}
