// A goto to the first statement of a do-while skips its condition, so the
// copied body must check the invariant before the continue condition.
int nondet_int();
int main() {
  int x = 0;
  __ESBMC_loop_invariant(x >= 0 && x <= 10);
  do {
  top:
    x++;
    if (x == 5 && nondet_int()) { x = 50; goto top; }
  } while (x < 10);
  __ESBMC_assert(x <= 10, "x <= 10");
  return 0;
}
