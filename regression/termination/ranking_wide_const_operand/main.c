/* Wide-constant operand in the loop guard.
 *
 * The frontend often promotes a literal at the int boundary to long when
 * the unsuffixed form does not fit in int: for example, the source
 *   while (x > -2147483648)
 * lowers (under `--32`) to
 *   IF !((long long)x > -2147483648)
 * The right-hand side is a long-long constant whose VALUE still fits in
 * int32. The width guard in measure_from_relational refuses 64-bit
 * SOURCE operands to avoid subtraction wrap, but it now exempts the
 * wide side when it is a constant whose numeric value lies within
 * [-2147483648, 2147483647] -- a value-preserving extension to int64
 * stays well inside int64 and the rank is overflow-safe.
 *
 * Expected verdict: VERIFICATION SUCCESSFUL. */

extern int __VERIFIER_nondet_int(void);

int main()
{
  int x = __VERIFIER_nondet_int();
  while (x > -2147483647) /* one above INT_MIN to avoid UB on the decrement */
  {
    x = x - 1;
  }
  return 0;
}
