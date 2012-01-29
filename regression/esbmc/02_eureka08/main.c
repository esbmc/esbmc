int a[2];
main(){
  int i,j;
  i=0;
  while(i<2){
    a[i]=0;
    i=i+1;
  }
  j=0;
  while(j<2){
    a[j]=j+1;
    j=j+1;
  }
  if(a[1]==0){
    assert(0);
  } else {
    ;
  }
}
