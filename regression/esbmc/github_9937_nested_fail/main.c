// #9937: re-storing an out-of-bounds pointer after a write to a struct
// member inside the union made the path vacuous.
int main()
{
  union { int *p; struct { long x; int *b; } s; } u;
  int arr[2];
  u.p = arr + 100;
  u.s.b = 0;
  u.p = u.p;
  __ESBMC_assert(u.p != arr + 100, "p kept");
}
