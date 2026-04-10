/* struct_nested_field_pass:
 * Entity contains an embedded Inner struct (pos.x, pos.y) and a scalar flag.
 * Contract: ensures pos.x, pos.y, and flags are set correctly.
 * Body is correct.
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <stddef.h>

typedef struct
{
  int x;
  int y;
} Inner;

typedef struct
{
  Inner pos;
  int flags;
} Entity;

void set_entity(Entity *e, int x, int y, int f)
{
  __ESBMC_requires(e != NULL);
  __ESBMC_ensures(e->pos.x == x && e->pos.y == y);
  __ESBMC_ensures(e->flags == f);
  e->pos.x = x;
  e->pos.y = y;
  e->flags = f;
}

int main() { return 0; }
