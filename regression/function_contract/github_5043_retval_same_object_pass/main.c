// Regression for #5043: a relational comparison on __ESBMC_return_value in an
// ensures clause must not spuriously fail the auto-generated SAME-OBJECT check.
// The function returns p, so return_value and p are the same object and
// (return_value <= p + 8) is well-defined and true. Before the fix, the
// ensures' SAME-OBJECT safety check was left in the original function body
// referencing an unbound return_value, producing a spurious violation.
char *f(char *p)
{
  __ESBMC_requires(__ESBMC_is_fresh(p, 8));
  __ESBMC_ensures((char *)__ESBMC_return_value <= (char *)p + 8);
  return p;
}
