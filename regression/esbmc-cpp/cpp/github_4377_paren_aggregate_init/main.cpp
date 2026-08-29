// esbmc/esbmc#4377 (C++20): parenthesised aggregate initialisation used to
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
  assert(t.b == 0);
  return 0;
}
