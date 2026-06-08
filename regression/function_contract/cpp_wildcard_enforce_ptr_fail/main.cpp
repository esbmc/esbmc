// Companion fail case: ensures claims vec[1] but body returns vec[0].

#define SIZE 5

int fst(const int *vec)
{
  __ESBMC_requires(vec != nullptr);
  __ESBMC_assigns();
  __ESBMC_ensures(__ESBMC_return_value == vec[1]); // wrong: should be vec[0]
  return vec[0];
}

int main()
{
  int vec[SIZE] = {10, 11, 12, 13, 14};
  int res = fst(vec);
  return 0;
}
