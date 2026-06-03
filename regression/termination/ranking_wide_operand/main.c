/* Soundness guard: the ranking checker's difference measure is computed
 * as m = (int64)a - (int64)b. That extension is value-preserving, so for
 * operands up to 32 bits the subtraction provably fits int64. For 64-bit
 * operands, however, a - b can fall outside int64 and WRAP under modular
 * bitvector subtraction, which could make a non-decreasing or unbounded
 * measure spuriously satisfy the bounded/decrease obligations and yield an
 * unsound "terminates" certificate.
 *
 * measure_from_guard therefore refuses guards whose operands exceed 32
 * bits. This loop has a 64-bit guard (long long a > b) and increases a, so
 * it does NOT terminate; the checker must decline (no "Ranking function"
 * line) and control must fall through to the existing machinery, leaving
 * the result UNKNOWN. A regression that let 64-bit operands through could
 * print VERIFICATION SUCCESSFUL here via a wrapped measure.
 *
 * Expected verdict: VERIFICATION UNKNOWN. */

typedef long long i64;

extern i64 __VERIFIER_nondet_longlong(void);

int main()
{
  i64 a = __VERIFIER_nondet_longlong();
  i64 b = __VERIFIER_nondet_longlong();
  while (a > b)
    a = a + 1;
  return 0;
}
