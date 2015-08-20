int main()
{
  unsigned n;
  int a[n];

  __ESBMC_assume(n<100000 && n>10);

  a[0]=0;
  a[1]=1;
  a[2]=2;

  assert(a[0]==1);
}
