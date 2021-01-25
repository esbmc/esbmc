int i;

void f()
{
  assert(!(i>100));
}

int main()
{
  int j;
  i=j;
  
  if(i>100) f();
}
