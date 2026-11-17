/* The native body dispatcher must emit the same if-guard as the legacy
 * round-trip, which peels the condition's `!` instead of wrapping it (#6759). */
int y;

void f(int ok)
{
  if (!ok)
    y = 1;
}

int main(void)
{
  f(0);
  return 0;
}
