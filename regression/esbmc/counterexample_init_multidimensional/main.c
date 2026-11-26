int main(void)
{
  int obj[2][2];
  __ESBMC_init_object(&obj);
  __ESBMC_assume(
    obj[0][0] == 1 && obj[0][1] == 2 && obj[1][0] == 3 && obj[1][1] == 4);
  __ESBMC_assert(
    !(obj[0][0] == 1 && obj[0][1] == 2 && obj[1][0] == 3 && obj[1][1] == 4),
    "audit failure");
  return 0;
}
