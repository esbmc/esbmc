int main()
{
  int x;
  
  x=-1;
  x=x>>1;  
  assert(x==-1);
  
  x=-10;
  x=x>>10;
  assert(x==-1);
}
