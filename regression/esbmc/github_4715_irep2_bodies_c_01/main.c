// Exercises C-frontend constructs under --irep2-bodies (V.4.3, esbmc#4715):
// initialized declarations (do-while scope fix), statement-expression macros
// (GNU { } extension), do-while loops, for loops with C99 init declarations.

// GNU statement-expression macro: exercises sideeffect("statement_expression")
// migration path.
#define MAX(a, b) ({ int _a = (a), _b = (b); _a > _b ? _a : _b; })

int arr_sum(int *a, int n)
{
  int s = 0; // initialized decl — triggers 2-op code_decl path
  for (int i = 0; i < n; i++)
    s += a[i]; // C99 for-init decl — exercises decl-block flattening
  return s;
}

int factorial(int n)
{
  int r = 1;
  do
  {
    r *= n;
    n--;
  } while (n > 0); // do-while: exercises code_dowhile2t migration
  return r;
}

int main()
{
  // Initialized declarations must survive round-trip without premature DEAD.
  int arr[4] = {1, 2, 3, 4};
  __ESBMC_assert(arr_sum(arr, 4) == 10, "arr_sum(1..4)==10");
  __ESBMC_assert(factorial(4) == 24, "4! == 24");
  int m = MAX(7, 3);
  __ESBMC_assert(m == 7, "MAX(7,3)==7");
  return 0;
}
