// Variable template specialisations differing only in a member-pointer
// argument must not share one symbol.
#include <cassert>
template <class T>
struct traits;
struct A
{
  int b;
};
struct B
{
  long e;
};
template <>
struct traits<int A::*>
{
  static const int v = 1;
};
template <>
struct traits<long B::*>
{
  static const int v = 2;
};
template <class T>
int vt = traits<T>::v;
int main()
{
  assert(vt<int A::*> == 1);
  assert(vt<long B::*> == 2);
  return 0;
}
