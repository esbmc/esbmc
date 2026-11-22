// The interval analysis reads i <= INT_MAX - 1 off the overflow check on
// i + 1, so i keeps its lower bound 1 and i * 3 cannot overflow; likewise
// j >= INT_MIN + 1 off j - 1 keeps j's upper bound -1.
int __VERIFIER_nondet_int(void);

int main()
{
  int n = __VERIFIER_nondet_int();
  if (n < 0)
    return 0;
  int a = 0;
  for (int i = 1; i <= n / 3; i++)
    a = i * 3;
  for (int j = -1; j >= -(n / 3); j = j - 1)
    a = j * 3;
  return a;
}
