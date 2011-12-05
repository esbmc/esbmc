int a[10];
main(){
  int i,j;

  i=0;
  while(i<10){  
    if(i==5){
      i=a[i];
      L: assert(a[i]>0);
    } else {
      j=i;
    }
    i++;
  }
}

