int f(void) { return 0; }
int main()
{
  __ESBMC_init_object((void *)f);
  return 0;
}
