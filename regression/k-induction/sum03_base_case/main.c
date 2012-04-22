#define a (2)
int nondet_int();
unsigned int nondet_uint();
_Bool nondet_bool();

int main() { 
  int sn=0;
  unsigned int loop1=0, n1=nondet_uint();
  unsigned int i=0;

  //for(i=1; i<=n; i++) {
  while(loop1<n1){
    loop1++;
     	if (i<10) //first case
    sn = sn + a; //second case OK
    //if (i==4) sn=-10; //third case OK
    i++;
  }
  //__ESBMC_assume(i>n);
  assert(sn==a || sn == 0);
}
