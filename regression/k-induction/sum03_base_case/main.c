#define a (2)
unsigned int nondet_uint();
struct sumt {
  int sn;
} sum;

int main() { 
  int i, n=nondet_uint(), sn=0;
  for(i=1; i<=n; i++) {
     	if (i<10) //first case
    sum.sn = sum.sn + a; //second case OK
    //if (i==4) sn=-10; //third case OK
  }
  //__ESBMC_assume(i>n);
  assert(sum.sn==n*a || sum.sn == 0);
}
