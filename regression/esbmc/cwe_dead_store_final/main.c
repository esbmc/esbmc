// A final dead store: x is assigned and never read before it goes out of
// scope. This is the textbook CWE-563 shape and exercises the end-of-scope
// DEAD handling (no later read keeps the value live).
int main(void)
{
  int x = 6;
  return 0;
}
