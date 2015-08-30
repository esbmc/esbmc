
int nondet_int();

int main() {
  int i=0, x=0, y=0;
  int n=nondet_int();
  __ESBMC_assume(n>0);
  for(i=0; 1; i++)
  {
    assert(x==0);
  }
  assert(x!=0);
}

