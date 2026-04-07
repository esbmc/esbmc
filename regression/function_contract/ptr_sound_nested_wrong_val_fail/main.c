/* ptr_sound_nested_wrong_val_fail: (soundness)
 * ensures(c->data->val == 42) but body sets c->data->val = 41.
 * Off-by-one on a nested pointer dereference — must be VERIFICATION FAILED.
 */
#include <stddef.h>

typedef struct { int val; } Inner;
typedef struct { Inner *data; } Container;

void set_value(Container *c)
{
  __ESBMC_requires(c != NULL);
  __ESBMC_requires(c->data != NULL);
  __ESBMC_ensures(c->data->val == 42);

  c->data->val = 41; /* off by one */
}

int main()
{
  Inner inner = {0};
  Container cont;
  cont.data = &inner;
  set_value(&cont);
  return 0;
}
