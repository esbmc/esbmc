int main(void)
{
  int x = 0, i = 0;
  if (1)
  {
    x = 0;
  }
  while (i++ < 2)
    ++x;
  do
    ++x;
  while (i++ < 4);
  if (x == 4)
    ++x;
  else
    x += 2;
  if (0)
    x += 10;
  else
    ++x;
  __ESBMC_assert(x == 7, "control bodies");
  return 0;
}
