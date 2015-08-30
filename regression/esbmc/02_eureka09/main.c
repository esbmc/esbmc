int a[5];
int max;

findmax(int dim){
  int i;
  i=0;
  while(i<dim){
    if(a[i]>max){
      max=a[i];
    } 
    i=i+1;
  }
}

main(){
  max=0;
  a[0]=1;
  a[1]=2;
  a[2]=3;
  a[3]=4;
  a[4]=5;
  findmax(5);
  ;
  if(max==0){
  ERROR: ;
  } 
}

