// A reference function the LD program's own file does not declare, standing
// in for the via-C side of a translation-validation check: this
// test pins that --secondary-entry-point keeps both entry points separate
// in one invocation, not that the two sides agree on anything.
int ref_entry(void)
{
  return 0;
}
