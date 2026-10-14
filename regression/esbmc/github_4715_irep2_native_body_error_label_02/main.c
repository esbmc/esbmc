// The error-label assertion is user_provided, which --no-assertions honours
// (symex_main.cpp skips user-provided claims under it). Nothing else in the
// suite pairs --error-label with --no-assertions, so without this the
// user_provided(true) call the native arm reproduces is untested: dropping it
// flips this test to FAILED and no other test notices.
extern int nd(void);
int main(void)
{
  if (nd())
    goto ERR;
  return 0;
ERR:;
}
