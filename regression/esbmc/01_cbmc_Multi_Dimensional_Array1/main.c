unsigned int a[10][4];
unsigned int *p;

int main()
{
  //assert(sizeof(a[0])==sizeof(int)*4);

  p=a[9];

  *p=1;
  
//  assert(a[9][0]==1);
  
//  p++;
  
  //*p=2;
  
//  assert(a[9][1]==2);
}
