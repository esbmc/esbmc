#define a (2)
int nondet_int();
unsigned int nondet_uint();
_Bool nondet_bool();

int main() { 
  int sn=0;
  unsigned int loop1=nondet_uint(), n1=nondet_uint();
  unsigned int x=0;

  //for(i=1; i<=n; i++) {
  while(1){
    //loop1++;
     	if (x<10) //first case
    sn = sn + a; //second case OK
    //if (i==4) sn=-10; //third case OK
    x++;
    assert(sn==x*a || sn == 0);
  }
  //__ESBMC_assume(i>n);
  
}
