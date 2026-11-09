// (c ? i : j) = v may set i, so it must forget i's lower bound 1: i * 3
// overflows on the sixth iteration.
int __VERIFIER_nondet_int();

int main()
{
  int n = __VERIFIER_nondet_int();
  if (n < 0)
    return 0;
  int a = 0, i = 1, j = 1;
  int c = __VERIFIER_nondet_int();
  while (i <= n / 3)
  {
    a = i * 3;
    if (i == 5)
      (c ? i : j) = -2147483647;
    else
      i = i + 1;
  }
  return a;
}
