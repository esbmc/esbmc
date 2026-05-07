// Regression for #4330 review. Companion to k_path_cov_9: pins the
// baseline witness count at the default --k-path-witness-depth=8 so a
// future regression that lowered the default silently (and thereby
// inflated apparent coverage by reducing the denominator) would fail
// here. Same source as k_path_cov_9 to make the contrast explicit:
// 14 witnesses at default depth, 2 at depth=2.
int main()
{
  int a, b, c;
  if (a > 0)
    ;
  else
    ;
  if (b > 0)
    ;
  else
    ;
  if (c > 0)
    ;
  else
    ;
}
