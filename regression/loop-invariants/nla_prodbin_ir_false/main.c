// Reduced from SV-COMP nla-digbench-scaling/prodbin-ll_unwindbound1.c
// (unreach-call expected_verdict: FALSE -- the loop is capped at 1 iteration,
// so z == a*b cannot yet hold and reach_error() is reachable).
// The in-loop relation z + x*y == a*b is a genuine loop invariant, annotated
// here as __ESBMC_loop_invariant. Under --loop-invariant-check --ir the loop
// summary abstracts the iteration cap and (with Int-sort arithmetic) discharges
// the post-loop assertion -> wrong VERIFICATION SUCCESSFUL without this fix.
extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__noreturn__));
void reach_error() { __assert_fail("0", "prodbin.c", 1, "reach_error"); }
extern int __VERIFIER_nondet_int(void);
void assume_abort_if_not(int c) { if (!c) abort(); }
void __VERIFIER_assert(int cond) { if (!cond) { ERROR: {reach_error();} } }

int counter = 0;
int main() {
  int a = __VERIFIER_nondet_int();
  int b = __VERIFIER_nondet_int();
  assume_abort_if_not(b >= 1);
  long long x = a, y = b, z = 0;

  __ESBMC_loop_invariant(z + x * y == (long long) a * b);
  while (counter++ < 1) {
    if (!(y != 0)) break;
    if (y % 2 == 1) { z = z + x; y = y - 1; }
    x = 2 * x; y = y / 2;
  }
  __VERIFIER_assert(z == (long long) a * b);
  return 0;
}
