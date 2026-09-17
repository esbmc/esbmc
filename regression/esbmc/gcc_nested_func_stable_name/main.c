/* The lifted helper has internal linkage, so its symbol id carries the basename
   of the file the transform wrote. That name must not vary between runs. */
int main(void)
{
  int base = 40;
  int add(int v)
  {
    return base + v;
  }
  __ESBMC_assert(add(2) == 42, "the lifted nested function is called");
  return 0;
}
