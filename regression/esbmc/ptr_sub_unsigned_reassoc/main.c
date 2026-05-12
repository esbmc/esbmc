/*
 * (p - u) + i - i must reassociate to p - u, NOT p + (-u_modular).
 * For unsigned u, neg2t lowers to (2^width - u) % 2^width — a large
 * positive value. If reassoc materializes the subtraction as
 * add(pointer, neg(unsigned)), the SMT pointer-arith conversion sees
 * the large positive value and moves the pointer forward by ~UINT_MAX.
 *
 * Regression for the bug Codex flagged in the rebuild_chain pointer-
 * chain path (commit 4f7c312069 family). Fix preserves negative offsets
 * as sub2tc(pointer, acc, term) for pointer chains.
 */
char buf[100];
int nondet_int();

int main() {
  unsigned u = 5;
  int i = nondet_int();
  __ESBMC_assume(i == 3);
  char *p = &buf[10];
  // Independent +i and -i force reassoc to fire and cancel them.
  // The remaining subtraction by `u` (unsigned) must stay subtractive.
  char *q = (p - u) + i - i;
  // q should point to &buf[5] (10 - 5 = 5).
  __ESBMC_assert(q == &buf[5], "(p - unsigned) + i - i preserves subtraction");
  return 0;
}
