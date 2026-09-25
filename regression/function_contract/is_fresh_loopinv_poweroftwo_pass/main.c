/* is_fresh_loopinv_poweroftwo_pass (issue #6399):
 * A wrapper loops K times over an __ESBMC_is_fresh struct, calling an
 * is_fresh-guarded callee (replaced via --replace-call-with-contract) per
 * element, with its own loop cut by __ESBMC_loop_invariant. This used to
 * report a false VERIFICATION FAILED whenever sizeof(Vec) was a power of two
 * (K = 1,2,4,8,16 -> 4,8,16,32,64 bytes). size_to_bit_width() gave an
 * n-element array ceil(log2(n)) index bits, which for a power-of-two n is
 * exactly log2(n): the one-past-the-end index n was then unrepresentable and
 * wrapped to 0, aliasing element 0. Every other size had a spare bit, hence
 * the parity. The domain is now wide enough to hold n itself.
 * sizeof(Vec) = 4*4 = 16 (power of two) here, the historically-failing shape.
 * The contract is genuinely true.
 */
typedef struct { int x; } Elem;
typedef struct { Elem vec[4]; } Vec;

void touch(Elem *e)
{
  __ESBMC_requires(__ESBMC_is_fresh(e, sizeof(Elem)));
  __ESBMC_ensures(e->x == 0);
  __ESBMC_assigns(e->x);
  e->x = 0;
}

void touch_all(Vec *v)
{
  unsigned k, j;
  __ESBMC_requires(__ESBMC_is_fresh(v, sizeof(Vec)));
  __ESBMC_assigns(v->vec);
  __ESBMC_ensures(__ESBMC_forall(&j, !(j < 4) || v->vec[j].x == 0));

  __ESBMC_loop_invariant(k <= 4 && __ESBMC_forall(&j, !(j < k) || v->vec[j].x == 0));
  for (k = 0; k < 4; k++)
    touch(&v->vec[k]);
}

int main(void) { return 0; }
