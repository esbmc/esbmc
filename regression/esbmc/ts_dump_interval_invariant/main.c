// The interval bound on x at the head is an invariant; the entry condition
// x < 10 is not, since it is false in the step that exits.
extern int __VERIFIER_nondet_int(void);
int main() {
  int x = 0;
  while (x < 10)
    if (__VERIFIER_nondet_int())
      x = x + 1;
  __ESBMC_assert(x == 10, "x");
  return 0;
}
