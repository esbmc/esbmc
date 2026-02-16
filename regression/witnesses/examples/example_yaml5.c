void reach_error()
{
}

int foo(int n)
{
  return n - 1;
}

int main()
{
  int x = 0;
  int r = foo(x);

  if (r < 0)
    reach_error();

  return 0;
}
