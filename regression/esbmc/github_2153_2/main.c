// Variant of github_2153: byte-manipulation via struct-field pointer cast
// at a non-zero byte offset.  After ptr[1] = nondet_char(), bug.bar can
// represent any pointer value, so the assertion must fail.

struct
{
  void *bar;
} bug;

int some_var;

int main()
{
  char *ptr = (char *)&bug.bar;
  ptr[1] = nondet_char();

  __ESBMC_assert(bug.bar != &some_var, "");
}
