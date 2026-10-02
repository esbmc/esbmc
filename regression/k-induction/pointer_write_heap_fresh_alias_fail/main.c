// p holds &x or an object the loop allocates. Unless the havoced p may still
// be &x, the inductive step drops the write and proves x == 0.
#include <stdlib.h>
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int x, y;

int main()
{
  int *p = &x;
  int i = 0;
  for (;;)
  {
    x = 0;
    if (i >= 5)
      *p = 5;
    __VERIFIER_assert(x == 0);
    if (i == 5)
      p = malloc(sizeof(int));
    i++;
  }
}
