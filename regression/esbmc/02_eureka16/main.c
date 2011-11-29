int a[2];
int i,j;

p(){
  int tmp;
  tmp=a[i];
  a[i]=a[j];
  a[j]=tmp;
}

main(){
  i=0;
  j=1;
  a[i]=100;
  a[j]=200;
  p();
  ;
  p();
  ;
  p();
  ;
  if(a[i]==100){
    assert(0);    
  } else {
    ;
  }
}

