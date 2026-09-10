// The control for array_element_destructors: the same class, the same count of
// objects, declared separately rather than as an array. These are destroyed
// correctly, which localises the defect to array elements.
#include <cassert>

int c = 0, d = 0;

struct M
{
  M()
  {
    ++c;
  }
  ~M()
  {
    ++d;
  }
};

int main()
{
  {
    M a;
    M b;
    M z;
  }
  assert(c == 3 && d == 3);
  return 0;
}
