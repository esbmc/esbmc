// Companion to github_5043_retval_same_object_pass: the ensures relational
// comparison on __ESBMC_return_value must still be genuinely enforced. Here the
// function returns p + 4, so (return_value <= p) is false and the contract is
// violated. This confirms the fix removes only the spurious safety check from
// the original body and does not weaken real ensures enforcement in the wrapper
// (return_value is correctly bound to the returned value, p + 4).
char *f(char *p)
{
  __ESBMC_requires(__ESBMC_is_fresh(p, 8));
  __ESBMC_ensures((char *)__ESBMC_return_value <= (char *)p);
  return p + 4;
}
