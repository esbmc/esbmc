// github #6464 (negative): the branch *is* taken and the write really is out of
// bounds of the object the pointer ends up at (aborts under
// clang -fsanitize=address). This pins end-to-end detection over the same
// merged-pointer shape as the positive tests; the claim it trips comes from
// bounds_check via build_reference_to, not from the guarded site itself.
#include <stdlib.h>

struct Elem
{
  char a[8];
};

int main()
{
  struct Elem *buf;
  int n = 0, cap = 0;

  buf = (struct Elem *)malloc(sizeof(struct Elem) * 10);
  if (!buf)
    abort();

  if (n == cap) // taken
  {
    struct Elem *nb = (struct Elem *)malloc(sizeof(struct Elem) * 1);
    if (!nb)
      abort();
    buf = nb;
    n = 5;
  }

  struct Elem e;
  e.a[0] = 1;
  *(buf + n) = e;
  return 0;
}
