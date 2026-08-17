#include<assert.h>

int A() { return 1; }

struct X {
  int a;
  int (*update) ();
};

int main()
{
  struct X x;
  x.update = A;

  int one = x.update();
  assert(one == 1);
  return 0;
}
