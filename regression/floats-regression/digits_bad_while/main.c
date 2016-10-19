extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }

int main()
{
  double x = 1.0/7.0;
  long long res = 0;

  int i = 1;
  while(x != 0)
  {
    res += ((int)(x * 10) % 10) * (i * 10);
    x = (x * 10) - (int) x * 10;
    i++;
  }

  __VERIFIER_assert(res > 56430);
  return 0;
}

