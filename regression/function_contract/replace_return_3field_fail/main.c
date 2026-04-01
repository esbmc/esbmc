/* replace_return_3field_fail:
 * Same contract.  Caller asserts v.z == 99 but ensures says z == 3.
 * The replace-call ASSUME(z == 3) constrains the return struct,
 * so assert(v.z == 99) is correctly reported as FAILED.
 *
 * Expected: VERIFICATION FAILED
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
  assert(v.z == 99); /* wrong: ensures guarantees z == 3 */
  return 0;
}
