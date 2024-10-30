int foo(int a, int b)
{
  if(a>0)
    printf("a>0\n");
  else if(b == 2)
    printf("b == 2\n");
  return a + b;
}

int main()
{
  int x = nondet_int();
  int y = nondet_int();
  assert(foo(x,y) == 10);
  return 0;
}