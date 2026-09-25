// github #6464: the bounds check for a pointer reconstructed from a byte array
// was emitted without the caller's same-object guard, so a second heap object
// reachable only on a never-taken branch made the access unprovable. Clean
// under clang -fsanitize=address,undefined.
//
// --no-propagation is load-bearing: with propagation on, the merge folds away
// before the check is built and the defect is invisible. Do not drop the flag.
#include <stdlib.h>

struct Elem
{
  char a[8];
};

int main()
{
  struct Elem *buf;
  int n = 0, cap = 10;

  buf = (struct Elem *)malloc(sizeof(struct Elem) * 10);
  if (!buf)
    abort();

  if (n == cap) // never taken
  {
    struct Elem *nb = (struct Elem *)malloc(sizeof(struct Elem) * 20);
    if (!nb)
      abort();
    buf = nb;
  }

  struct Elem e;
  e.a[0] = 1;
  *(buf + n) = e;
  return 0;
}
