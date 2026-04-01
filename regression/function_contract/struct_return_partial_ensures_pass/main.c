/* struct_return_partial_ensures_pass:
 * Documents expected behaviour when ensures only covers SOME fields.
 * Vec3 has fields x, y, z.  Contract ensures only x and y; z is unconstrained.
 * Body deliberately sets z = z_param + 999 (wrong), but since z is NOT in
 * the ensures, verification passes — the contract is incomplete by design.
 *
 * This is a SOUNDNESS DOCUMENTATION test: incomplete ensures cannot catch
 * bugs in unconstrained fields.  Callers that rely on z being z_param will
 * receive no contract guarantee.
 *
 * Expected: VERIFICATION SUCCESSFUL (z violation is outside the contract)
 */
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
  /* z intentionally NOT in ensures */
  Vec3 v;
  v.x = x;
  v.y = y;
  v.z = z + 999; /* BUG in z, but not covered by contract */
  return v;
}

int main() { return 0; }
