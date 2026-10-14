/* The companion to github_7267-const-guard: suppressing constant-folded
   guards must not stop a genuinely unreachable branch being reported. */
int f(void)
{
  return 1;
}

int main(void)
{
  int x = 5;
  if (x > 10)
    f();
  return 0;
}
