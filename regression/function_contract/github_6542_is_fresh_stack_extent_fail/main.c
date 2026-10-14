/* github_6542_is_fresh_stack_extent_fail:
 * Accepting a named object must not mean accepting any extent it is asked for.
 * DYNAMIC_SIZE holds nothing for automatic storage, so the extent comes from
 * the object's type instead; 16 bytes cannot discharge a request for 4096. */
typedef struct { int coeffs[4]; } P;

void callee(P *p) {
  __ESBMC_requires(__ESBMC_is_fresh(p, 4096));
  __ESBMC_assigns(p->coeffs);
  __ESBMC_ensures(p->coeffs[0] == 1);
  p->coeffs[0] = 1;
}

int main(void) {
  P v;
  callee(&v);
  return 0;
}
