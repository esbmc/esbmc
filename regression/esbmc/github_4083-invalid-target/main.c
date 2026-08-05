// Jumping to a value that is not any label's address is undefined behaviour.
// The dispatch chain is exhaustive over the address-taken labels, so the
// trailing assertion is what reports it instead of silently falling through
// to the next statement (issue #4083).
int main()
{
  void *p = (void *)0;
  void *unused[] = {&&L1};
  goto *p;
L1:
  return 0;
}
