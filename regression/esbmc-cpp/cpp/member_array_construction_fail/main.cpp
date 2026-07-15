// Negative counterpart: every element of the member array is constructed with
// i == 1, so asserting a later element differs must fail.
#include <cassert>

struct B
{
  int i;
  B() : i(1)
  {
  }
};

struct A
{
  B b_array[3];
  A()
  {
  }
};

int main()
{
  A a;
  assert(a.b_array[2].i == 2);
  return 0;
}
