// A goto to the head of a loop headed by an interval ASSUME starts a new
// iteration, so the copied body must check the invariant there.
int nondet_int();
int main() {
  int x = 0;
  __ESBMC_loop_invariant(x >= 0 && x <= 10);
L:
  while (x < 10) {
    x++;
    if (x == 5 && nondet_int()) {
      x = 7;
      goto L;
    }
  }
  __ESBMC_assert(x <= 10, "x <= 10");
  return 0;
}
