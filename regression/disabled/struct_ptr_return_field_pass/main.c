/* struct_ptr_return_field_pass:
 * Function returns Point* with field-level ensures.
 * Contract: if non-null, both x and y must match the inputs.
 * Body correctly assigns x and y — satisfies the ensures.
 *
 * Tests the fix for pointer-return field access:
 *   ((T*)__ESBMC_return_value)->field  (no &, no address_of)
 *
 * Expected: VERIFICATION SUCCESSFUL
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
    p->x = x;
    p->y = y;
  }
  return p;
}

int main() { return 0; }
