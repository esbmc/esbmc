/* struct_ptr_return_nonnull_fail:
 * Function returns Point* with ensures(return != NULL).
 * Body returns NULL unconditionally — violates the non-null ensures.
 *
 * Expected: VERIFICATION FAILED
 */
#include <stddef.h>

typedef struct
{
  int x;
  int y;
} Point;

Point *alloc_point(int x, int y)
{
  __ESBMC_requires(x >= 0 && y >= 0);
  __ESBMC_ensures(__ESBMC_return_value != (Point *)0);
  return (Point *)0; /* BUG: always returns NULL */
}

int main() { return 0; }
