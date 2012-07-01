
int nondet_int();

int main() {
  int i=0, x=0, y=0;
  int n=nondet_int();
  __ESBMC_assume(n>0);
  for(i=0; i<n; i++)
  {
    x = x-y;
    assert(x==0);
    y = nondet_int();
    __ESBMC_assume(y!=0);
    x = x+y;
    assert(x!=0);
  }
  //__ESBMC_assume(i>=n);
  assert(x==0);
}

