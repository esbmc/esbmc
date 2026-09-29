// #8055
int ext(void);

int main()
{
  int n = 0, i = 0;
  n = ext();
  while (i < n)
  {
    i++;
    if (i > 3)
      break;
  }
  __ESBMC_assert(i == 0, "loop not entered");
}
