/* struct_return_3field_pass:
 * Struct returned by value with 3 fields.  Contract covers all three.
 * Body is correct.
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <stddef.h>

typedef struct { int x; int y; int z; } Vec3;

Vec3 make_vec3(int x, int y, int z)
{
  __ESBMC_requires(x >= 0 && y >= 0 && z >= 0);
  __ESBMC_ensures(((Vec3 *)&__ESBMC_return_value)->x == x);
  __ESBMC_ensures(((Vec3 *)&__ESBMC_return_value)->y == y);
  __ESBMC_ensures(((Vec3 *)&__ESBMC_return_value)->z == z);
  Vec3 v;
  v.x = x;
  v.y = y;
  v.z = z;
  return v;
}

int main() { return 0; }
