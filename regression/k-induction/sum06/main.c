#define a (2)
int nondet_int();

int main()
{
  int i=1, n = nondet_int(), sn = 0;
  __VERIFIER_assume(n < 1000 && n >= -1000);

  while (i <= n)
  {
    sn = sn + a;
    i++;
    assert(sn == (i-1) * a);
  }

  assert(sn == n * a || sn == 0);
}
