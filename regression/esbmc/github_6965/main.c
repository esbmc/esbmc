extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *);
void reach_error(void) { __assert_fail("0", "main.c", 4, "reach_error"); }
void __VERIFIER_assert(int c) { if (!c) { reach_error(); abort(); } }

/* An uninterpreted function is a function of its arguments. ESBMC cannot give
   the SMT backend a tuple-sorted UF symbol, so pointer-argument applications
   are Ackermannised in symex instead of losing congruence (#6965). */
_Bool __CPROVER_uninterpreted_equals(const void *const a, const void *const b);
unsigned long __CPROVER_uninterpreted_hasher(const void *const a);

int main(void)
{
  int x, y;
  const void *a = &x, *b = &y;

  __VERIFIER_assert(
    __CPROVER_uninterpreted_equals(a, b) == __CPROVER_uninterpreted_equals(a, b));
  __VERIFIER_assert(
    __CPROVER_uninterpreted_hasher(a) == __CPROVER_uninterpreted_hasher(a));
  return 0;
}
