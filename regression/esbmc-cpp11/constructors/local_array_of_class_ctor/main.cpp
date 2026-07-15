#include <cassert>

// Regression for local (automatic-storage) arrays of class type: every element
// must be constructed, not just element 0 (see Constructor9-1).  Also covers:
//  - directly-declared multidimensional local arrays (per-element recursion),
//  - member arrays,
//  - aggregate initialisation (must keep each element's own arguments),
//  - function-local static arrays (constructed once by static_lifetime_init,
//    NOT re-constructed from the function body).

int g;

struct B
{
  int i;
  B() : i(1) { g++; }        // default constructor
  B(int x) : i(x) { g++; }   // value constructor
};

struct Wrap
{
  B b[2];                    // member array
};

void with_static()
{
  static B s[2];             // function-local static: constructed once, ever
  assert(s[0].i == 1);
  assert(s[1].i == 1);
}

int main()
{
  // Whole-array default construction.
  B def[3];
  assert(def[0].i == 1 && def[1].i == 1 && def[2].i == 1);

  // Directly-declared multidimensional local array: all four leaves.
  B twod[2][2];
  assert(twod[0][0].i == 1 && twod[0][1].i == 1);
  assert(twod[1][0].i == 1 && twod[1][1].i == 1);

  // Member array default construction.
  Wrap w;
  assert(w.b[0].i == 1 && w.b[1].i == 1);

  // Aggregate initialisation must keep each element's own arguments.
  B agg[2] = { B(7), B(8) };
  assert(agg[0].i == 7 && agg[1].i == 8);

  // Function-local static array constructed exactly once across both calls.
  with_static();
  with_static();

  // Constructors ran once per element: def 3 + twod 4 + w.b 2 + agg 2
  // + static 2 = 13.
  assert(g == 13);

  return 0;
}
