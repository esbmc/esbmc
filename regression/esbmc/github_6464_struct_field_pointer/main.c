// github #6464: same defect reached through a pointer held in a struct field,
// which --incremental-bmc exposes because it does not constant-propagate WITH
// chains. Clean under clang -fsanitize=address,undefined.
#include <stdlib.h>

struct Elem
{
  char a[8];
};

struct Holder
{
  struct Elem *buf;
  int n;
  int cap;
};

int main()
{
  struct Holder h;

  h.buf = (struct Elem *)malloc(sizeof(struct Elem) * 10);
  if (!h.buf)
    abort();
  h.n = 0;
  h.cap = 10;

  if (h.n == h.cap) // never taken
  {
    struct Elem *nb = (struct Elem *)malloc(sizeof(struct Elem) * 20);
    if (!nb)
      abort();
    h.buf = nb;
    h.cap = 20;
  }

  struct Elem e;
  e.a[0] = 1;
  *(h.buf + h.n) = e;
  return 0;
}
