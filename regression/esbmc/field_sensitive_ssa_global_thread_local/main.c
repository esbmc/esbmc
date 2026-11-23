#include <assert.h>

struct S
{
  int a;
  int b;
};

_Thread_local struct S t;

int main(void)
{
  struct S l = {2, 3};
  t.a = l.a;
  assert(t.a == 2 && t.b == 0 && l.b == 3);
  return 0;
}
