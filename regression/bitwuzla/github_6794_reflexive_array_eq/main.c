/* array_conv encodes an array equality index-by-index, so a frame check that
 * compares an array against its own snapshot -- the same array id at the same
 * update level -- indexed a valuation vector that is empty whenever no select
 * was ever applied to that array, and aborted (#6794). Reached via the tuple
 * flattener, so every backend registering no tuple_api takes this path. */
#include <stdint.h>
#include <stddef.h>
#define N 8
typedef struct
{
  int16_t coeffs[N];
} poly;
void add(poly *r, const poly *b)
{
  __ESBMC_requires(r != NULL && b != NULL);
  __ESBMC_assigns(r->coeffs);
  __ESBMC_ensures(1);
  for (unsigned i = 0; i < N; i++)
    r->coeffs[i] = (int16_t)(r->coeffs[i] + b->coeffs[i]);
}
