// Without slicing the summarised array reaches the solver whole.
#define N 100
int a[N];
int main()
{
  for (int i = 0; i < N; i++)
    a[i] = i;
  for (int x = 0; x < N; x++)
    __ESBMC_assert(a[x] == x, "a[x] == x");
}
