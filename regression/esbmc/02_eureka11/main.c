int i,j;
int a[4];
main(){
  a[0]=0;
  a[3]=3;
  j=0;
  while(j<2){
    a[j]=a[3-j];
    j=j+1;
  }
  if(a[0]==0){
    assert(0);
  }
}
