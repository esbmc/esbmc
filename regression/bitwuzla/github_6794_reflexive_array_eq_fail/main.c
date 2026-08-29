/* Negative companion to github_6794_reflexive_array_eq: the reflexive equality
 * must be discharged as true without discharging the frame check around it, so
 * the write to b->coeffs[0] -- absent from the assigns clause -- is still
 * caught. Pins the fix as sound rather than merely crash-avoiding (#6794). */
#include <stdint.h>
#include <stddef.h>
#define N 8
typedef struct
{
  int16_t coeffs[N];
} poly;
void add(poly *r, poly *b)
{
  __ESBMC_requires(r != NULL && b != NULL);
  __ESBMC_assigns(r->coeffs);
  __ESBMC_ensures(1);
  for (unsigned i = 0; i < N; i++)
    r->coeffs[i] = (int16_t)(r->coeffs[i] + b->coeffs[i]);
  b->coeffs[0] = 42;
}
