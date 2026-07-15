#include <assert.h>
int main()
{
  struct S
  {
    int a;
    int b;
  } s;
  s.a = 3;
  s.b = 4;
  assert(s.a + s.b == 8);
  return 0;
}