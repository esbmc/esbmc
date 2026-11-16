// #9907: the pointer sits in a nested union of a member smaller than the array
#include <assert.h>
#include <stdlib.h>
union U {
  int z[3];
  struct { int a; union { char c; int *cur; } n; } ov;
};
int main(void)
{
  int x = 5;
  union U w;
  w.ov.n.cur = &x;
  union U *p = &w;
  assert(*p->ov.n.cur == 5);
  return 0;
}
