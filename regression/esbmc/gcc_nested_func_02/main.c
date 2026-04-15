// Nested function reads enclosing variable
int main()
{
  int val = 7;
  int get()
  {
    return val;
  }
  int r = get();
  __ESBMC_assert(r == 7, "nested func reads enclosing var");
  return 0;
}
