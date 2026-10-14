// Failing companion to github_4715_irep2_bodies_vla_01.
//
// Same VLA-dimension side-effect construct `b[++a]` migrated under
// --irep2-bodies (exercises the migrate_expr side-effect path that used to
// abort), but with a property that is violated so the verdict is FAILED.
// Confirms ON and OFF agree on the FAILED verdict, not just on not crashing.
//
// Expected: VERIFICATION FAILED on both paths.

void foo(char **q)
{
  q[5];
}

int main()
{
  int a = 5;
  char *b[++a];                            // a becomes 6
  __ESBMC_assert(a == 5, "a is 6 after ++a, so this fails");
  foo(b);
  return 0;
}
