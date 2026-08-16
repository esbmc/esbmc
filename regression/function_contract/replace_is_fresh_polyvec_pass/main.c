/* replace_is_fresh_polyvec_pass: the real-world motivating pattern for #6380 --
 * a "vector of objects" wrapper, itself requiring the whole vector be fresh,
 * that calls a per-element function on each element in a loop, where the
 * per-element function's own contract requires the element be fresh.
 *
 * This mirrors mlkem-native's mlk_polyvec_reduce looping over MLKEM_K
 * polynomials calling mlk_poly_reduce once per element. Before #6380 the
 * poly-level is_fresh precondition could not be discharged under contract
 * replacement (it failed vacuously), forcing a rewrite to != NULL and losing
 * the non-aliasing guarantee. With the assert-side lowering, is_fresh(&v->vec[k])
 * on the fresh vector's elements verifies. Expected: SUCCESSFUL.
 */
#define K 3
#define N 4
typedef struct { int coeffs[N]; } poly;
typedef struct { poly vec[K]; } polyvec;

void poly_reduce(poly *r)
{
  __ESBMC_requires(__ESBMC_is_fresh(r, sizeof(poly)));
  __ESBMC_assigns(r->coeffs);
  __ESBMC_ensures(r->coeffs[0] == 0);
  for (int i = 0; i < N; i++)
    r->coeffs[i] = 0;
}

void polyvec_reduce(polyvec *v)
{
  __ESBMC_requires(__ESBMC_is_fresh(v, sizeof(polyvec)));
  __ESBMC_assigns(v->vec);
  __ESBMC_ensures(v->vec[0].coeffs[0] == 0);
  for (int k = 0; k < K; k++)
    poly_reduce(&v->vec[k]);
}

int main(void) { return 0; }
