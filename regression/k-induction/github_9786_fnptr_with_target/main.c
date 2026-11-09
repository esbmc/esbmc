/* #9786: a call through a pointer that has a target still disables the
 * inductive step. */
int cb(int x)
{
  return x;
}

int (*fp)(int) = cb;

int main(void)
{
  fp(1);
  unsigned i = 0;
  while (1)
  {
    i = (i + 1) % 4;
    __ESBMC_assert(i < 4, "i stays below 4");
  }
  return 0;
}
