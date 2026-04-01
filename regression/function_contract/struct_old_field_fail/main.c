/* struct_old_field_fail:
 * Same contract.  Body increments count by 2 instead of 1.
 * ensures(c->count == old(c->count) + 1) catches the wrong delta.
 *
 * Expected: VERIFICATION FAILED
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
  c->count += 2; /* BUG: should be +1 */
}

int main() { return 0; }
