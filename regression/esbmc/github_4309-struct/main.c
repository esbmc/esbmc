#include <assert.h>

// Stress test: nondet whole-struct, assertion on its fields.
// Confirms that struct-field nondets are surfaced in the Inputs line
// (or, if not, exposes a gap in collect_nondet_values for non-scalar
// nondets). The assertion is violated by multiple distinct (x, y)
// tuples on the boundary.

struct point
{
  int x;
  int y;
};

int main(void)
{
  struct point p;
  if (p.x < -1 || p.x > 1)
    return 0;
  if (p.y < -1 || p.y > 1)
    return 0;
  // Violated when |x| + |y| <= 1, i.e. (0,0), (1,0), (-1,0), (0,1),
  // (0,-1) — five witnesses if struct fields are independently
  // nondet.
  int absx = p.x < 0 ? -p.x : p.x;
  int absy = p.y < 0 ? -p.y : p.y;
  assert(absx + absy > 1);
  return 0;
}
