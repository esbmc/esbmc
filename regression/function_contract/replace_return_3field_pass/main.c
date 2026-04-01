/* replace_return_3field_pass:
 * --replace-call-with-contract with a 3-field struct-return function.
 * After replacement, the caller's assertion on all three fields must hold
 * because the ensures constraints propagate through the return value.
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <assert.h>
#include <stddef.h>

typedef struct
{
  int x;
  int y;
  int z;
} Vec3;

Vec3 make_vec3(int x, int y, int z)
{
  __ESBMC_requires(x >= 0 && y >= 0 && z >= 0);
  __ESBMC_ensures(((Vec3 *)&__ESBMC_return_value)->x == x);
  __ESBMC_ensures(((Vec3 *)&__ESBMC_return_value)->y == y);
  __ESBMC_ensures(((Vec3 *)&__ESBMC_return_value)->z == z);
  Vec3 v = {x, y, z};
  return v;
}

int main()
{
  Vec3 v = make_vec3(1, 2, 3);
  assert(v.x == 1 && v.y == 2 && v.z == 3);
  return 0;
}
