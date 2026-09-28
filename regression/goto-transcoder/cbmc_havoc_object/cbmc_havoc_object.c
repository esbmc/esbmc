#include <assert.h>

void __CPROVER_havoc_object(void *);

struct S
{
  int x;
  int y;
};

int main(void)
{
  int a[3] = {1, 2, 3};
  __CPROVER_havoc_object(a);
  a[1] = 7;
  assert(a[1] == 7);

  struct S s = {1, 2};
  __CPROVER_havoc_object(&s.x);
  s.x = 4;
  s.y = 5;
  assert(s.x + s.y == 9);

  int v = 8;
  __CPROVER_havoc_object(&v);
  v = 6;
  assert(v == 6);

  return 0;
}
