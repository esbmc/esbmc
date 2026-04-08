/* struct_old_field_pass:
 * __ESBMC_old() applied to individual fields of a pointed-to struct.
 * Both val and count are incremented by exactly 1.
 * ensures check delta via old().
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <stddef.h>

typedef struct
{
  int val;
  int count;
} Counter;

void increment(Counter *c)
{
  __ESBMC_requires(c != NULL);
  __ESBMC_ensures(c->val == __ESBMC_old(c->val) + 1);
  __ESBMC_ensures(c->count == __ESBMC_old(c->count) + 1);
  c->val++;
  c->count++;
}

int main() { return 0; }
