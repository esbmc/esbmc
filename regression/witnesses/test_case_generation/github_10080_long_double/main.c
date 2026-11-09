/* #10080: a binary128 long double needs 36 digits to read back as itself. */
extern void reach_error(void);
extern long double __VERIFIER_nondet_long_double(void);
int main()
{
  long double x = __VERIFIER_nondet_long_double();
  long double lo = 0.1L;
  if (x > lo && x < lo + 0x1p-115L)
    reach_error();
  return 0;
}
