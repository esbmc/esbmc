unsigned int nondet_uint();

int main()
{
  unsigned int n = nondet_uint();
//  __ESBMC_assume(n>0 && n<10000);
  unsigned int x=n, y=0;
//  __ESBMC_assume(x==n);
  __ESBMC_loop_invariant(x + y == n);
  while(x>0)
  {
    x--;
    y++;
  }
  assert(y!=n);
//  assert(x==0);
}
