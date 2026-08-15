_Bool flag = 0;

int main()
{
  flag = nondet_bool();
  if (!flag)
  {
    flag = 1;
    __ESBMC_assert(!flag, "stale interval fact must not survive the write");
  }
  return 0;
}
