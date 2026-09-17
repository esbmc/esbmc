struct s
{
  int arr[4];
};

int main(void)
{
  struct s v;
  struct s *ptr = &v;

  /* An array-typed operand decays before the comparison; without that the two
     sides are an array and a pointer. */
  __ESBMC_assert(ptr->arr != 0, "an array-typed member decays to a pointer");
  return 0;
}
