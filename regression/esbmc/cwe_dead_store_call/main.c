extern int f(void);

// The value returned by f() is stored in x and never read. The call still
// executes (its side effects are preserved); only the unused assignment of the
// return value is a dead store (CWE-563). Pins call-return site handling.
int main(void)
{
  int x = f();
  return 0;
}
