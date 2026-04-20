// C++ variant: --enforce-contract '*' --function fst with a pointer parameter.
// Exercises:
//   1. is_compiler_generated now correctly ignores "c:@F@fst#*1I#" (has '#'
//      from C++ USR parameter encoding, but is NOT compiler-generated).
//   2. find_function_symbol("fst") falls back to a short-name search and finds
//      the symbol whose full ID is "c:@F@fst#*1I#".
//   3. alloc_ptr_params comparison uses func_sym->name ("fst") so the wildcard
//      path gets backing storage for the pointer parameter.

#define SIZE 5

int fst(const int *vec)
{
  __ESBMC_requires(vec != nullptr);
  __ESBMC_assigns();
  __ESBMC_ensures(__ESBMC_return_value == vec[0]);
  return vec[0];
}

int main()
{
  int vec[SIZE] = {10, 11, 12, 13, 14};
  int res = fst(vec);
  return 0;
}
