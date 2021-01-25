int global;

void f(int i)
{
  global=1;
}

void g(int i)
{
  global=0;
}

int main()
{
  void (*p)(int);
  _Bool c;
  
  p=c?f:g;
  
  p(1);
  
  assert(global!=c);
}
