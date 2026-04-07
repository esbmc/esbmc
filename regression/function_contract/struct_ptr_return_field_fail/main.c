/* struct_ptr_return_field_fail:
 * Function returns Point* with field-level ensures.
 * Contract: if non-null, both x and y must match the inputs.
 * Body sets p->x = 0 instead of x — violates the ensures when non-null.
 *
 * Tests the fix for pointer-return field access:
 *   ((T*)__ESBMC_return_value)->field  (no &, no address_of)
 *
 * Expected: VERIFICATION FAILED
 */
#include <stdlib.h>
#include <stddef.h>

typedef struct
{
  int x;
  int y;
} Point;

Point *make_point(int x, int y)
{
  __ESBMC_requires(x >= 0 && y >= 0);
  /* When non-null, the fields must match the parameters. */
  __ESBMC_ensures(
    __ESBMC_return_value == (Point *)0 ||
    (((Point *)__ESBMC_return_value)->x == x &&
     ((Point *)__ESBMC_return_value)->y == y));
  Point *p = (Point *)malloc(sizeof(Point));
  if (p)
  {
    p->x = 0; /* BUG: should be x */
    p->y = y;
  }
  return p;
}

int main() { return 0; }
