// Positive half of github_4715_irep2_native_body_vla_retype_01_fail: the
// snapshotted bound must still admit an in-range access after the size variable
// is reassigned.
int main(void)
{
  int n = 8;
  int a[n];
  a[0] = 42;
  n = 1;
  return a[5];
}
