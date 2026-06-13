// Issue #5224 negative control: a loop that modifies only a scalar (no
// pointer write) must NOT trip the pointer-write gate, so the inductive
// step stays enabled and proves this unbounded-in-the-base-case program.
// Matching "Solution found by the inductive step" confirms the gate did
// not fire (a spurious gate would force VERIFICATION UNKNOWN instead).
extern int nondet_int(void);

int main(void)
{
  int n = nondet_int();
  __ESBMC_assume(n > 0);

  int x = 0;
  for (int i = 0; i < n; i++)
    x = 0;

  __ESBMC_assert(x == 0, "x stays zero across all iterations");
  return 0;
}
