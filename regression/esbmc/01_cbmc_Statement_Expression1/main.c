int main()
{
  int x;
  int y;

  // as a side-effect  
  ({ x=1; x;});
  
  assert(x==1);
  
  x= ({ y=1; 2; });

  assert(x==2);
  assert(y==1);

  return 0;
}
