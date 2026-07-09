// Reduced from SV-COMP nla-digbench-scaling/egcd-ll_unwindbound5.c
// (unreach-call expected_verdict: FALSE -- loop capped at 5 iterations).
// The three in-loop relations are a genuine mutually-inductive invariant,
// annotated as __ESBMC_loop_invariant. --loop-invariant-check --ir wrongly
// reports SUCCESSFUL without this fix.
extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__noreturn__));
void reach_error() { __assert_fail("0", "egcd.c", 1, "reach_error"); }
extern int __VERIFIER_nondet_int(void);
void assume_abort_if_not(int c) { if (!c) abort(); }
void __VERIFIER_assert(int cond) { if (!cond) { ERROR: {reach_error();} } }

int counter = 0;
int main() {
  long long a, b, p, q, r, s;
  int x = __VERIFIER_nondet_int();
  int y = __VERIFIER_nondet_int();
  assume_abort_if_not(x >= 1);
  assume_abort_if_not(y >= 1);
  a = x; b = y; p = 1; q = 0; r = 0; s = 1;

  __ESBMC_loop_invariant(1 == p * s - r * q);
  __ESBMC_loop_invariant(a == y * r + x * p);
  __ESBMC_loop_invariant(b == x * q + y * s);
  while (counter++ < 5) {
    if (!(a != b)) break;
    if (a > b) { a = a - b; p = p - q; r = r - s; }
    else       { b = b - a; q = q - p; s = s - r; }
  }
  __VERIFIER_assert(a - b == 0);
  return 0;
}
