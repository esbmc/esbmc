// i overflows only after 2^31 iterations: the inductive step must not
// prove it safe, and the base case cannot reach the overflow.
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int i = 0;
  while (__VERIFIER_nondet_int())
    i = i + 1;
  return 0;
}
