// Reduced from SV-COMP nla-digbench-scaling/geo2-ll_unwindbound5.c
// (unreach-call expected_verdict: TRUE). Constant-bounded loop with a post-loop
// assertion: this fix skips the summary under --ir, but k-induction/BMC re-proves
// the small bound, so the correct SUCCESSFUL verdict is preserved (guards against
// the fix over-approximating away a safe task).
extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__noreturn__));
void reach_error() { __assert_fail("0", "geo2.c", 1, "reach_error"); }
extern int __VERIFIER_nondet_int(void);
void __VERIFIER_assert(int cond) { if (!cond) { ERROR: {reach_error();} } }

int counter = 0;
int main() {
  int z = __VERIFIER_nondet_int();
  int k = __VERIFIER_nondet_int();
  unsigned long long x = 1, y = 1, c = 1;

  __ESBMC_loop_invariant(1 + x * z - x - z * y == 0);
  while (counter++ < 5) {
    if (!(c < k)) break;
    c = c + 1; x = x * z + 1; y = y * z;
  }
  __VERIFIER_assert(1 + x * z - x - z * y == 0);
  return 0;
}
