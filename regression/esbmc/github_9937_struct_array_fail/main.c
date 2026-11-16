// #9937: re-storing an array element that holds an out-of-bounds pointer
// made a claim on the pointer hold vacuously.
int main()
{
  union { int *p; struct { int *a; } s[1]; } u;
  int arr[2];
  u.s[0].a = arr + 100;
  u.s[0] = u.s[0];
  __ESBMC_assert(u.p != arr + 100, "p kept");
}
