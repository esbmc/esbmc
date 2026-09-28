/* github_6542_is_fresh_stack_interior_extent_fail:
 * An interior pointer into an automatic object has room only for what follows
 * it. &v.coeffs[2] sits 8 bytes into a 16-byte object, so 8 bytes remain and a
 * request for 16 must be refused. Paired with
 * github_6542_is_fresh_stack_interior_extent_pass, which asks for the 8 that
 * are there. */
typedef struct { int coeffs[4]; } P;

void callee(int *p) {
  __ESBMC_requires(__ESBMC_is_fresh(p, 16));
  __ESBMC_assigns(p[0]);
  __ESBMC_ensures(p[0] == 1);
  p[0] = 1;
}

int main(void) {
  P v;
  callee(&v.coeffs[2]);
  return 0;
}
