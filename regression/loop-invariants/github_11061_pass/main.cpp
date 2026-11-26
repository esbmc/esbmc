// https://github.com/esbmc/esbmc/issues/11061
bool small(int i)
{
  return i <= 3;
}

int main()
{
  int idx = 0;
  __ESBMC_loop_invariant((idx == 0) || small(idx));
  __ESBMC_loop_invariant(0 <= idx && idx <= 3);
  while (idx < 3)
  {
    ++idx;
  }
}
