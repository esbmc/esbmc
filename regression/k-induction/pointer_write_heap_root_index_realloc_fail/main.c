// realloc copies B's address, stored as an integer, into d's new object, so
// what the loop loads from it reaches B. Unless the inductive step havocs B's
// object, B[1] keeps its pre-loop 0 and the step proves B[1] < 3.
#include <stdlib.h>
#include <assert.h>
extern int nondet_int(void);
int main()
{
  int *B = malloc(2 * sizeof(int));
  B[0] = 0;
  B[1] = 0;
  long *d0 = malloc(sizeof(long));
  *d0 = (long)B;
  long *d = realloc(d0, sizeof(long));
  while (nondet_int())
  {
    int *r = (int *)(*d + 4);
    *r = *r + 1;
    assert(B[1] < 3);
  }
  return 0;
}
