/* Companion to regression/bitwuzla/github_6794_reflexive_array_eq, pinning the
 * same fix on a z3-only build. array_conv is reached through the TUPLE
 * flattener: a backend registering no tuple_api gets smt_tuple_node_flattener,
 * which owns an array_convt. z3 does register one, so --tuple-node-flattener is
 * what puts this path back in reach here (#6794). */
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
