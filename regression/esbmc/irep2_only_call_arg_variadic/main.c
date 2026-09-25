/* A variadic argument has no parameter type to convert against, so only the
   array decay is owed -- to void *. */
int vfun(int n, ...);

int main(void)
{
  int a[3];
  a[0] = 1;
  return vfun(1, a);
}
