/* struct_return_ptr_field_pass:
 * Struct contains a pointer field (data) and a scalar (size).
 * Contract: ensures both fields match the arguments.
 * Body is correct.
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <stddef.h>

typedef struct
{
  int *data;
  int size;
} Buf;

void init_buf(Buf *b, int *d, int sz)
{
  /* Guarded: an unconditional is_fresh would also state b apart from d, and
   * this contract states no such separation. The _fail twin matches. */
  __ESBMC_requires(b == NULL || __ESBMC_is_fresh(b, sizeof(Buf)));
  __ESBMC_requires(b != NULL && d != NULL && sz > 0);
  __ESBMC_ensures(b->data == d);
  __ESBMC_ensures(b->size == sz);
  b->data = d;
  b->size = sz;
}

int main() { return 0; }
