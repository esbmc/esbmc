int a[10];
int main(){
  int b[2];
  int i;
  i=0;
  while(i<10){
    a[i]=i+1;
    i=i+1;
  }
  if(a[8]==0){
    assert(0);
  } else {
    ;
  }
  return 0;
}
