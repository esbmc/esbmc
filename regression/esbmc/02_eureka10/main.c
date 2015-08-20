int a[5];

swap(int x,int y){
  int tmp;
  tmp=a[x];
  a[x]=a[y];
  a[y]=tmp;
}

main(){
  int j,i;
  a[0]=0;
  a[1]=1;
  a[2]=2;
  a[3]=3;
  a[4]=4;
  j=0;
  i=4;
  while(i>=0){
    while(j<i){
      if(a[j]<=a[j+1]){
	swap(j,j+1);
	;
      } else {
	;
      }
      j=j+1;
    }
    i=i-1;
  }
  if(a[4]==0){
    assert(0);
  } 
}
