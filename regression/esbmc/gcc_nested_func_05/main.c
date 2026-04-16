// Two levels of nesting
int main()
{
  int x = 0;
  void outer()
  {
    void inner()
    {
      x = 99;
    }
    inner();
  }
  outer();
  __ESBMC_assert(x == 99, "doubly nested function");
  return 0;
}
