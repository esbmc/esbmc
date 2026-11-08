/* #10080: the only double in the branch is 0x3FB999999999999B. */
extern void reach_error(void);
extern double __VERIFIER_nondet_double(void);
int main()
{
  double x = __VERIFIER_nondet_double();
  if (x > 0.1 && x < 0.10000000000000003)
    reach_error();
  return 0;
}
