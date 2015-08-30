#include <assert.h>

struct A
{
  unsigned int i;
  _Bool j;
};

struct B
{
  unsigned int i;
  _Bool j;
  int k;
};

int func(struct A a)
{
  return a.i;
}

int main()
{
  struct B b;
//  int x;

  b.i = 1;
  b.j = 1;
  assert((* ((struct A*)&b)).i == 1); // This works fine.
  assert((* ((struct A*)&b)).j == 1); // This works fine.

//  x=func( * ((struct A*)&b));
//  assert(x == 1);
}
