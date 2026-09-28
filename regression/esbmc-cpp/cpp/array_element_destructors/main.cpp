// KNOWNBUG. Destructors of array elements never run: ESBMC constructs all
// three objects and destroys none of them (c == 3 && d == 0 verifies here,
// c == 3 && d == 3 does not). g++ runs three of each.
//
// array_element_destructors_control shows the same three objects declared
// separately are destroyed correctly, so it is the array, not the class.
// array_element_destructors_leak shows what it costs a user: an element
// holding a heap allocation is reported as leaking, because the destructor
// that frees it is skipped.
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
    M a[3];
  }
  assert(c == 3 && d == 3);
  return 0;
}
