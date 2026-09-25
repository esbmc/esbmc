int foo(int x) { return 1; }
int main(void)
{
  int il;
  for (il = 0; foo(il) && il < 2; ++il)
  {
  }
  return 0;
}
