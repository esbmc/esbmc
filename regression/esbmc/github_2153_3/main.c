// Variant of github_2153: byte-manipulation through a char* cast to the whole
// struct (not to a specific field).  The goto-symex produces a struct-level
// byte_update, so bug.bar is extracted from byte_update<struct>(...).
// After ptr[0] = nondet_char(), bug.bar can represent any pointer value.

struct
{
  void *bar;
} bug;

int some_var;

int main()
{
  char *ptr = (char *)&bug; /* points to the struct, not to bug.bar directly */
  ptr[0] = nondet_char();

  __ESBMC_assert(bug.bar != &some_var, "");
}
