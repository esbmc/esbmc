#include <assert.h>

struct A
{
  int i;
  int j;
};

struct B
{
  int i;
  int j;
  int k;
};

int func(struct A a)
{
  return a.i;
}

int main()
{
  struct B b;
  int x;

  b.i = 1;
  b.j = 1;
  assert((* ((struct A*)&b)).j == 2); // This works fine.

  x=func( * ((struct A*)&b));
  assert(x == 1);
}
