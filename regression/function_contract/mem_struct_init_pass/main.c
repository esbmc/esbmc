/* mem_struct_init_pass:
 * Contract for a struct initialiser: ensures every field is set to
 * the caller-supplied values.  Body is correct.
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <stddef.h>

typedef struct { int x; int y; int z; } Vec3;

void vec3_init(Vec3 *v, int x, int y, int z)
{
  __ESBMC_requires(v != NULL);
  __ESBMC_ensures(v->x == x && v->y == y && v->z == z);
  v->x = x;
  v->y = y;
  v->z = z;
}

int main() { return 0; }
