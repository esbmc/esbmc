// A lambda's call operator is converted while the enclosing body is still
// mid-conversion. The address-taken label set is per-function state, so the
// nested conversion must stack it rather than reset it -- otherwise main's
// later `&&A` resolves against the lambda's (empty) set and the whole
// function fails to convert (issue #4083).
int main()
{
  void *p[2];
  int r = 0;
  auto inner = [](int v) { return v + 3; };
  r = inner(4);
  p[0] = &&A;
  p[1] = &&B;
  goto *p[1];
A:
  r += 1;
  goto END;
B:
  r += 2;
  goto END;
END:
  __ESBMC_assert(r == 9, "lambda conversion preserves the outer label set");
  return 0;
}
