

int main()
{
  int non_zero_array[10];
  int sym;
  __ESBMC_assert(
    __ESBMC_forall(&sym, !(sym >= 0 && sym < 10) || non_zero_array[sym] == 0),
    "array is zero initialized");
}
