// #9937: the path of github_9937_array is reachable; master made it vacuous
// when the array was read back byte by byte.
int main()
{
  union { int *p[2]; struct { int *a[2]; long x; } s; } u;
  int arr[2], brr[2];
  u.p[0] = arr + 100;
  u.p[1] = brr + 50;
  u.s.x = 7;
  __ESBMC_assert(u.p[0] == arr + 100, "p[0] kept");
  __ESBMC_assert(u.p[1] == brr + 50, "p[1] kept");
  __ESBMC_assert(0, "reachable");
}
