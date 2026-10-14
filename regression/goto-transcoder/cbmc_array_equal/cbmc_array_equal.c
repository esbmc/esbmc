int nondet_int(void);

int main(void)
{
  int a[3] = {1, 2, 3};
  int b[3] = {1, 2, 3};
  __CPROVER_assert(__CPROVER_array_equal(a, b), "equal");

  int c[3] = {1, 2, 9};
  __CPROVER_assert(!__CPROVER_array_equal(a, c), "differ in the last element");

  // Symbolic contents: the result tracks the values, not the syntax.
  int p[2], q[2];
  p[0] = nondet_int();
  p[1] = nondet_int();
  q[0] = p[0];
  q[1] = p[1];
  __CPROVER_assert(__CPROVER_array_equal(p, q), "symbolic equal");

  int r[2];
  r[0] = p[0];
  r[1] = ~p[1];
  __CPROVER_assert(!__CPROVER_array_equal(p, r), "symbolic differ");

  char s[2] = {'x', 'y'};
  char t[2] = {'x', 'y'};
  __CPROVER_assert(__CPROVER_array_equal(s, t), "char equal");

  return 0;
}
