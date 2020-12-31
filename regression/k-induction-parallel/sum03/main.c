#define a (2)
int nondet_int();
unsigned int nondet_uint();
_Bool nondet_bool();

int main() { 
  int sn=0;
  unsigned int x=0;

  while(1){
    sn = sn + a;
    x++;
    assert(sn==x*a || sn == 0);
  }
}
