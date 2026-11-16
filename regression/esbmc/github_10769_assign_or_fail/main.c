// An || whose operands call functions, assigned rather than branched on.
extern int __VERIFIER_nondet_int(void);
extern void reach_error(void);
int neg(int x) { return x < 0; }
int big(int x) { return x > 100; }
int main(void)
{
  int x = __VERIFIER_nondet_int();
  int out = neg(x) ||
            big(x);
  if (out)
    reach_error();
  return 0;
}
