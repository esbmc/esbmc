#include <cassert>

struct S
{
  int a;
  int b;
};

int main()
{
  // Parenthesised aggregate init binds 1 to .a and 2 to .b; the assertion
  // below must fail.
  S s(1, 2);
  assert(s.a == 9);
  return 0;
}
