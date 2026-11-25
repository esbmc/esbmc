extern void reach_error(void);
int main(void)
{
  __ESBMC_assert(0, "dereference failure: unrelated application condition");
  reach_error();
}
