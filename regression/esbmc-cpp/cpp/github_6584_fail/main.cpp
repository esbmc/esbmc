// Negative twin: the constructor really runs for every element, so a claim
// that a later element is unset must be reported rather than passing.
#include <cassert>

struct T
{
  int v;
  T() : v(7)
  {
  }
};

int main()
{
  T *p = new T[3];
  assert(p[2].v != 7);
  delete[] p;
  return 0;
}
