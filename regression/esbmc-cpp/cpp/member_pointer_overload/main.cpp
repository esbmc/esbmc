// Overloads differing only in a member-pointer parameter must not share one
// symbol: clang's USR spells that parameter type as nothing.
#include <cassert>
struct A
{
  int b;
};
struct B
{
  long e;
};
int f(int A::*)
{
  return 1;
}
int f(long B::*)
{
  return 2;
}
int main()
{
  assert(f(&A::b) == 1);
  assert(f(&B::e) == 2);
  return 0;
}
