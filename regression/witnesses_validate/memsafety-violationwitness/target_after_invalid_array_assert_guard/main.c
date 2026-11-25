extern void reach_error(void);
int main(void)
{
  int a[1] = {0};
  int i = 1;
  __ESBMC_assert(a[i] == 0, "application condition");
  reach_error();
}
