// Clang's __builtin_ forms of the string/allocation functions are rewritten to
// the names ESBMC models. memset, memcmp, strncpy and calloc were missing from
// that list, so each was left unmodelled and its effect went nondet -- a
// __builtin_memset'd struct kept garbage fields, which surfaced as a spurious
// "dereference failure: invalid pointer" on
// gcc.c-torture/execute/20000815-1.c.
struct S
{
  int a;
  int b;
};

int main(void)
{
  struct S e;
  __builtin_memset(&e, 0, sizeof(e));
  __ESBMC_assert(e.a == 0 && e.b == 0, "__builtin_memset zeroes the struct");

  char x[4] = "abc", y[4] = "abc";
  __ESBMC_assert(__builtin_memcmp(x, y, 4) == 0, "__builtin_memcmp compares");

  char d[8];
  __builtin_strncpy(d, "hi", 3);
  __ESBMC_assert(d[0] == 'h' && d[2] == '\0', "__builtin_strncpy copies");

  int *p = __builtin_calloc(2, sizeof(int));
  if (p)
    __ESBMC_assert(p[0] == 0 && p[1] == 0, "__builtin_calloc zeroes");
  return 0;
}
