int *num;

int main()
{
  void *data_rgn_base = malloc(*num);
  __ESBMC_init_object(data_rgn_base);
  return 0;
}
