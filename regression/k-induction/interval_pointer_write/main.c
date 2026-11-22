// The write through p cannot change i, whose address is never taken, so i
// keeps its lower bound 1 and i * 3 cannot overflow.
#include <stdlib.h>
int __VERIFIER_nondet_int(void);

int main()
{
  int n = __VERIFIER_nondet_int();
  if (n < 0)
    return 0;
  int *p = malloc(sizeof(int));
  if (!p)
    return 0;
  for (int i = 1; i <= n / 3; i++)
    *p = i * 3;
  free(p);
  return 0;
}
