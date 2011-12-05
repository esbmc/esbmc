main() {
    int i;
    int a[3];
    
    a[1] = 0;
    i = 0;
    while((a[i]!=0)&&(i<3)){
      a[i] = 2*i;
      i = i+1;
    }
    if(i>=2){
      assert(0);
    } else {
      ;
    }
}
