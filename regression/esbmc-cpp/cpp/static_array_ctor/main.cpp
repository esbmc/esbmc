// A static/global array of a class type with a non-trivial constructor must
// have every element constructed before main() runs (C++ [basic.start.static]
// + [class.init]).  ESBMC used to treat the array as an aggregate and drop the
// CXXConstructExpr entirely, so no element constructor ran.
#include <cassert>

int constructed;

struct Counter
{
  int tag;
  int untouched;
  Counter() : tag(42)
  {
    ++constructed;
  }
};

// 1-D and 2-D static arrays: 3 + 2*2 = 7 constructor calls in total.
static Counter row[3];
static Counter grid[2][2];

int main()
{
  assert(constructed == 7);

  // Every element is fully constructed ...
  assert(row[0].tag == 42 && row[2].tag == 42);
  assert(grid[0][0].tag == 42 && grid[1][1].tag == 42);

  // ... and members the constructor never assigns are zero-initialised.
  assert(row[1].untouched == 0);
  assert(grid[0][1].untouched == 0);
  return 0;
}
