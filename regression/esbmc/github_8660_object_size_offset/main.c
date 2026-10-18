#include <assert.h>
#include <stdlib.h>
int main(void)
{
  char buf[10];
  struct { int a; char s[8]; } x;
  struct { char s[4]; int tail; } y;
  char *p = buf + 4;
  char *q = &x.s[2];
  char *r = &y.s[1];
  assert(__builtin_object_size(p, 0) == 6);
  assert(__builtin_object_size(p, 2) == 6);
  assert(__builtin_object_size(q, 0) == 6);
  assert(__builtin_object_size(q, 1) == 6);
  assert(__builtin_object_size(r, 0) == 7);
  assert(__builtin_object_size(buf + 12, 0) == 0);
  char *h = malloc(16);
  if (h)
    assert(__builtin_object_size(h + 5, 0) == 11);
  return 0;
}
