// #2356: building the counterexample under --cvc5 --ir read an integer model
// value as a bit-vector and aborted.
int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x + 2 == -3);
  __ESBMC_assert(x != -5, "x is -5");
  return 0;
}
