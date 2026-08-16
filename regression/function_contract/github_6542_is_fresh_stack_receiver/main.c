/* github_6542_is_fresh_stack_receiver:
 * A caller may pass the address of a live automatic object (#6380).
 * __ESBMC_alloc is written for heap objects only, so VALID_OBJECT(&v) was a
 * free boolean a solver could pick false, and no stack caller could discharge
 * the precondition. */
typedef struct { int coeffs[4]; } P;

void callee(P *p) {
  __ESBMC_requires(__ESBMC_is_fresh(p, sizeof(P)));
  __ESBMC_assigns(p->coeffs);
  __ESBMC_ensures(p->coeffs[0] == 1);
  p->coeffs[0] = 1;
}

int main(void) {
  P v;
  callee(&v);
  return 0;
}
