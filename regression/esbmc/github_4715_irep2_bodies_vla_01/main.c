// Regression under --irep2-bodies (esbmc#4715, V.4.4 verdict parity);
// reproducer adapted from regression/esbmc/github_178_unary.
//
// A side-effect (here a preincrement) used as a VLA dimension `b[++a]` is part
// of the array TYPE. Under --irep2-bodies the body — and the types it mentions
// — are migrated to IREP2 before goto_convert lowers the VLA. The preincrement
// side-effect therefore reaches migrate_expr, which used to unconditionally
// read the side-effect's "#size" sub for every non-allocation statement. A
// preincrement has no "#size", so migrating the empty exprt aborted with
// "migrate expr failed". The flag-OFF path never hit this (the VLA is lowered
// before migration), so OFF reported SUCCESSFUL while ON crashed.
//
// Expected: VERIFICATION SUCCESSFUL on both paths.

void foo(char **q)
{
  q[5]; // unused expression statement: index nested in a code_expression
}

int main()
{
  int a = 5;
  char *b[++a]; // VLA dimension is a preincrement side-effect
  __ESBMC_assert(a == 6, "++a increments a to 6");
  foo(b); // b has 6 elements; q[5] is in bounds
  return 0;
}
