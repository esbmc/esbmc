#define a (2)
int nondet_int();
int main()
{
  int i, n = nondet_int(), sn = 0;
  for (i = 1; i <= n; i++)
    sn = sn + a;

  assert(sn == n * a || sn == 0);
}
