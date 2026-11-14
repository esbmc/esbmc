/* #10642: the invariant admits x == 1, which never occurs, and from there
 * reach_error() is reachable. Other invariant states skip it, so the
 * violation is not forced by the invariant and the claim stays unknown. */
extern void reach_error(void);
extern int __VERIFIER_nondet_int(void);
int main() {
  int x = 0, y = 0;
  __ESBMC_loop_invariant(0 <= x && x <= 1 && 0 <= y && y <= 1);
  while (__VERIFIER_nondet_int()) {
    if (x == 1)
      y = 1;
    if (y == 1)
      reach_error();
    x = 0;
  }
  return 0;
}
