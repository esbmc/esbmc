int arr[2] = {0,0};

int main()
{
  unsigned place = nondet_int();
  __ESBMC_assume(place<2);
  assert(arr[place]==0);
  return 0;
}
