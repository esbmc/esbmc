/* struct_nested_field_fail:
 * Same contract.  Body sets flags = f + 1 (wrong).
 * ensures(e->flags == f) catches the violation even though pos is correct.
 *
 * Expected: VERIFICATION FAILED
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
  e->flags = f + 1; /* BUG: flags off by one */
}

int main() { return 0; }
