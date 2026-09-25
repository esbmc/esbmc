// The declaration and its bool test live in the loop body, but the loop stays
// a for-loop so continue still reaches the increment. If continue jumped to
// the re-declaration instead, j would never decrease and the loop would not
// terminate (issue #4078).
struct c
{
  int a;
  c(int x) : a(x)
  {
  }
  operator bool()
  {
    return a != 0;
  }
};

int main()
{
  int n = 0;
  for (int j = 3; c b = j; j--)
  {
    if (j == 2)
      continue;
    n++;
  }
  __ESBMC_assert(n == 2, "continue runs the increment and skips one body");
  return 0;
}
