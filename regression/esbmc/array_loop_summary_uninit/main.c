// The uninitialised-variable check would read the summary's unwritten
// elements, so the pass stays off under it.
int main()
{
  int a[4];
  for (int i = 0; i < 4; i++)
    a[i] = 0;
  __ESBMC_assert(a[2] == 0, "a[2]");
}
