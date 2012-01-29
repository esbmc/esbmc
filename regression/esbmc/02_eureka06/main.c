int i;
int a[20];

int nondet_int(void);

main(){
  i=0;
  i=nondet_int();
  if(i>0){
    while(i<20){
      a[i]=i;
      i=i+1;
    }
  } else {
  L: assert(0);
  }
}
