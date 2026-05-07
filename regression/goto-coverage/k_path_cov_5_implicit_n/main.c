// Regression for #4330 Copilot-1: `--k-path-coverage` without `=N` must
// produce a coverage report. Earlier the implicit-value sentinel (0) was
// stored verbatim, so `optionst::get_bool_option` (atoi) returned false
// and the report was silently dropped even though instrumentation ran.
int main()
{
  int a, b;
  if (a > 0)
    a = 1;
  else
    a = -1;

  if (b > 0)
    b = 1;
  else
    b = -1;

  return a + b;
}
