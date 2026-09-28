// esbmc/esbmc#4377 negative control (C++20): parenthesised aggregate initialisation used to
// abort with "Conversion of unsupported clang expr: CXXParenListInitExpr".
#include <cassert>

struct S
{
  int a;
  int b;
};

int main()
{
  S s(1, 2);
  assert(s.a == 1);
  assert(s.b == 2);

  S t(5);
  assert(t.a == 5);
  assert(t.b == 1);
  return 0;
}
