extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *);
void reach_error(void) { __assert_fail("0", "main.c", 4, "reach_error"); }
void __VERIFIER_assert(int c) { if (!c) { reach_error(); abort(); } }

/* Congruence must not collapse into "the result is fixed": distinct arguments
   stay unconstrained, so this assertion is still violable (#6965). */
_Bool __CPROVER_uninterpreted_equals(const void *const a, const void *const b);

int main(void)
{
  int x, y;
  const void *a = &x, *b = &y;

  __VERIFIER_assert(
    __CPROVER_uninterpreted_equals(a, b) == __CPROVER_uninterpreted_equals(b, a));
  return 0;
}
