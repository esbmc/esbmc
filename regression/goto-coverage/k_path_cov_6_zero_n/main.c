// Regression for #4330 Copilot-2: `--k-path-coverage=0` must be rejected
// at parse time, not silently fall back to --unwind/4. Same code as
// k_path_cov_1.
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
