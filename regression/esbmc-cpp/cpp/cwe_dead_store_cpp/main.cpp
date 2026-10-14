// Regression for a crash: --dead-store-check segfaulted on any C++ input
// because the linked C++ exception model contains a void call (terminate())
// whose return operand is a null expr, and the pass called is_symbol2t on it
// without a null guard. This trivial C++ program links that model; it must now
// run cleanly and still report the dead store of the overwritten x = 5.
int main()
{
  int x = 5;
  x = 6;
  return x;
}
