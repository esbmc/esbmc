void reach_error()
{
}

int foo(int n)
{
  if (n == 0)
  {
    reach_error();
  }
  return n + 1;
}

int main()
{
  int x = nondet_int();
  return foo(x);
}
