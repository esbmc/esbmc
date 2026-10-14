// Positive half of github_4715_irep2_native_body_error_label_01_fail: a label
// that does not match --error-label keeps the plain target, so the goto lands
// on the labelled statement and nothing is asserted.
extern int nd(void);
int main(void)
{
  if (nd())
    goto DONE;
  return 0;
DONE:;
}
