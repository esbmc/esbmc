/* An in-range conversion, and a symbolic one the caller has bounded, must not
   be flagged; truncation toward zero makes the representable operands exactly
   [MIN, MAX + 1) (#7572). */
extern double nondet_double(void);

int main(void)
{
  double d = 42.5;
  long long x = (long long)d;
  if (x != 42)
    return 1;

  double e = nondet_double();
  __ESBMC_assume(e >= -100.0 && e <= 100.0);
  int y = (int)e;
  (void)y;

  double lo = -2147483648.0;
  int z = (int)lo;
  (void)z;
  return 0;
}
