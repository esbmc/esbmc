// Negative variant of static_array_ctor: the total number of element
// constructions for a static array of a class with a non-trivial constructor
// is exactly the array size (3), so asserting a different count must fail.
#include <cassert>

int constructed;

struct Counter
{
  Counter()
  {
    ++constructed;
  }
};

static Counter row[3];

int main()
{
  // Only 3 elements exist, so this assertion is violated.
  assert(constructed == 2);
  return 0;
}
