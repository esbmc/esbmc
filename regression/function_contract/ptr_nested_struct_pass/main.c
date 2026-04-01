/* ptr_nested_struct_pass:
 * Tests contracts with a double-level pointer dereference: c->data->val.
 * The struct Container holds a pointer to an Inner struct.
 * requires(c->data != NULL) constrains the pointer field.
 * ensures(c->data->val == 42) verifies the nested write.
 * Caller sets up the concrete nested structure — no --assume-nonnull-valid.
 */
#include <assert.h>
#include <stddef.h>

typedef struct { int val; } Inner;
typedef struct { Inner *data; } Container;

void set_value(Container *c)
{
  __ESBMC_requires(c != NULL);
  __ESBMC_requires(c->data != NULL);
  __ESBMC_ensures(c->data->val == 42);

  c->data->val = 42;
}

int main()
{
  Inner inner = {0};
  Container cont;
  cont.data = &inner;
  set_value(&cont);
  assert(inner.val == 42);
  return 0;
}
