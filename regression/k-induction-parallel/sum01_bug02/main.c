#define a (2)
unsigned int nondet_uint();
int main() { 
  int i, j=10, n=nondet_uint(), sn=0;
  for(i=1; i<=n; i++) {
    if (i<j) //first case
    sn = sn + a; //second case OK
    j--;
    //if (i==4) sn=-10; //third case OK
  }
  //__ESBMC_assume(i>n);
  assert(sn==n*a || sn == 0);
}
