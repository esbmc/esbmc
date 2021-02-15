#define a (2)
#define SIZE 8
unsigned int nondet_uint();
int main() { 
  int i, sn=0;
  for(i=1; i<=SIZE; i++) {
    //if (i<4) //first case
    sn = sn + a; //second case OK
    //if (i==4) sn=-10; //third case OK
  }
  //__ESBMC_assume(i>n);
  assert(sn==SIZE*a || sn == 0);
}
