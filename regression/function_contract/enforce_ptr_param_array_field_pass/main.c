/* Regression: --enforce-contract on a function with a SECOND pointer
 * parameter whose pointee is a struct containing an ARRAY field used to abort
 * with "Can't construct rvalue reference to array type during dereference":
 * the Phase 2C frame check snapshotted the whole array (*b).coeffs by value.
 * Distilled from mlkem-native's mlk_poly_add(mlk_poly *r, const mlk_poly *b).
 * The function honours its assigns clause, so enforcement must succeed. */
#include <stdint.h>
#include <stddef.h>
#define N 8
typedef struct { int16_t coeffs[N]; } poly;
void add(poly *r, const poly *b)
{
  __ESBMC_requires(r != NULL && b != NULL);
  __ESBMC_assigns(r->coeffs);
  __ESBMC_ensures(1);
  for (unsigned i = 0; i < N; i++)
    r->coeffs[i] = (int16_t)(r->coeffs[i] + b->coeffs[i]);
}
