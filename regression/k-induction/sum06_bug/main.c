#define a 2
int nondet_int();

int main()
{
  unsigned long long int i=1, sn=0;
  int n = nondet_int();

  __VERIFIER_assume(n < 1000 && n >= -1000);

  while ( i <= n ) {
    sn = sn + ((i%10==9)? 4 : a);
    i++;
  }

  assert(sn == n * a || sn == 0);
}
