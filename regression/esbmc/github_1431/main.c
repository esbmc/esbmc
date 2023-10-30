__attribute__((annotate("__ESBMC_inf_size")))
_Bool __ESBMC_my_array[1];

int main()
{
  unsigned long int my_value;
  __ESBMC_my_array[my_value] = 1;
  return 0;
}
