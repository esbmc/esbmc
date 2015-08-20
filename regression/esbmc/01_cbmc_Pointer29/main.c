#include <assert.h>

struct A
{
  char a;
  unsigned int i;
  _Bool j;
};

struct B
{
  char a;
  unsigned int i;
  _Bool j;
  int k;
  double m;
};

int func(struct A a)
{
  return a.i;
}

int main()
{
  struct B b;
//  int x;

  b.a = 'a';
  b.i = 1;
  b.j = 1;

  assert((* ((struct A*)&b)).a == 'a'); // This works fine.
  assert((* ((struct A*)&b)).i == 1); // This works fine.
  assert((* ((struct A*)&b)).j == 1); // This works fine.

//  x=func( * ((struct A*)&b));
//  assert(x == 1);
}
