int more(void);
int main()
{
  int i = 0;
  while (({
    if (i > 3)
      goto out;
    more();
  }))
    ++i;
out:
  __ESBMC_assert(i <= 4, "the statement expression leaves at 4");
  return i;
}
