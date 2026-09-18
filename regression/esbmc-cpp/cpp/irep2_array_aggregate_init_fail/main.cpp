#include <cassert>

// Aggregate initialisation arrives as a constant_array of per-element
// initialisers, not as one whole-array constructor call, and must not be fanned
// out: every element would be constructed with element 0's arguments.
struct B
{
  int v;
  B(int p) : v(p)
  {
  }
};

int main()
{
  B a[2] = {B(1), B(2)};
  assert(a[1].v == 1);
}
