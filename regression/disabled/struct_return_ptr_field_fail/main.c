/* struct_return_ptr_field_fail:
 * Same struct with pointer field.  Body sets size = sz + 1 (wrong).
 * ensures(b->size == sz) catches the scalar-field violation even when
 * the pointer field is correct.
 *
 * Expected: VERIFICATION FAILED
 */
#include <stddef.h>

typedef struct
{
  int *data;
  int size;
} Buf;

void init_buf(Buf *b, int *d, int sz)
{
  __ESBMC_requires(b != NULL && d != NULL && sz > 0);
  __ESBMC_ensures(b->data == d);
  __ESBMC_ensures(b->size == sz);
  b->data = d;
  b->size = sz + 1; /* BUG: wrong size */
}

int main() { return 0; }
