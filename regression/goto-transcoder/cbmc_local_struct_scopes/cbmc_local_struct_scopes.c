#include <assert.h>
// Same struct name "S" with different layouts in two scopes. Scope-qualified
// type-symbol names must keep them distinct (f's S has one field, main's two).
void f(void)
{
  struct S
  {
    int a;
  } s;
  s.a = 1;
  assert(s.a == 1);
}
int main()
{
  struct S
  {
    int a;
    int b;
  } s;
  s.a = 2;
  s.b = 3;
  f();
  assert(s.a + s.b == 5);
  return 0;
}