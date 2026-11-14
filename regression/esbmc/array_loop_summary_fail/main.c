#define N 100000
int arr[N];
int main()
{
  for (int i = 0; i < N; i++)
    arr[i] = i;
  for (int i = 0; i < N; i++)
    if (i == 50000)
      arr[i] = 3;
  for (int i = 0; i < N; i++)
    __ESBMC_assert(arr[i] == i, "arr[i] == i");
}
