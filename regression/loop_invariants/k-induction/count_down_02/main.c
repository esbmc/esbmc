int nondet_int();
int main() { 

  int fail=0;
  int x=nondet_int();
  __ESBMC_assume(x>0);
  __ESBMC_loop_invariant(x >= 0);
  while(x>0) {
    __ESBMC_assume(fail!=1);
    if(fail==1) {
      x= -1;
//      fail=2;
    } else {
      x--;
    }
  }
  assert(x==0);
}

