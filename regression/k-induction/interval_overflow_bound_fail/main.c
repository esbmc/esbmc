// x + 1 overflows on the eighth iteration. The bound x <= INT_MAX - 1 taken
// from its overflow check must not let the inductive step prove it first.
int __VERIFIER_nondet_int(void);

int main()
{
  int x = 2147483640;
  while (__VERIFIER_nondet_int())
    x = x + 1;
  return x;
}
