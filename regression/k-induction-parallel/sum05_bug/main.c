#define a (2)
int nondet_int();
int main() { 
  int i, n=nondet_int(), sn=0;
  for(i=1; i<=n-2; i++) {
    //if (i<10) //first case
    sn = sn + a; //second case OK
    //if (i==4) sn=-10; //third case OK
  }
  //__ESBMC_assume(i>n);
  assert(sn==n*a || sn == 0);
}
