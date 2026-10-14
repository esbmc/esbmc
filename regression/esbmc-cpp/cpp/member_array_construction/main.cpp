// A class member that is an array of class type must have every element
// default-constructed, not just element 0. The member initialiser used to
// lower to a single `B(&a.b_array)` call (constructing only b_array[0]); it now
// constructs each element (bounded to modestly-sized arrays).
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
  B grid[2][2];
  A()
  {
  }
};

int main()
{
  A a;
  assert(a.b_array[0].i == 1);
  assert(a.b_array[1].i == 1);
  assert(a.b_array[2].i == 1);
  assert(a.grid[0][0].i == 1);
  assert(a.grid[1][1].i == 1);
  return 0;
}
