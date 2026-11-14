/* #10642's counterpart: the invariant and the exit condition force x == 3,
 * so every state leaving the loop reaches reach_error() and the violation is
 * real. */
extern void reach_error(void);
extern int __VERIFIER_nondet_int(void);
int main() {
  int x = 0;
  __ESBMC_loop_invariant(0 <= x && x <= 3);
  while (x < 3) {
    x++;
  }
  if (x == 3)
    reach_error();
  return 0;
}
