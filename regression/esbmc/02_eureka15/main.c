int nondet_int(void);

main(){
  int x,i;
  x=5;
  x=nondet_int();
  while(x<4){    
    if(x>0){
      x=x+1;  
    } else {
      x=1;
    }
  }
 L: assert(0);
}
