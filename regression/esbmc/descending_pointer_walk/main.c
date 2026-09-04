int main()
{
  int a[2][3];
  int steps = 0;
  for (int *p = &a[1][2]; p != &a[0][0]; p--)
    steps++;
  __ESBMC_assert(steps == 5, "five steps back to the first element");
  return 0;
}
