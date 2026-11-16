// #9937: copying the bits of an out-of-bounds pointer through an integer
// member made a claim on it hold vacuously.
int main()
{
  union { int *p[2]; long l; } u;
  int arr[2];
  u.p[0] = arr + 100;
  u.l = u.l;
  __ESBMC_assert(u.p[0] != arr + 100, "copied pointer");
}
