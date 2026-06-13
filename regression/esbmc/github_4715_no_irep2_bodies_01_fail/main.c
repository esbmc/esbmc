// Failing variant of the --no-irep2-bodies escape-hatch test: identical
// structured control flow, one deliberately wrong assertion, so the legacy
// body-lowering path must still report VERIFICATION FAILED.

int main()
{
  int x = 0;

  if (1)
    x = 1;

  while (x < 3)
    x++;

  for (int i = 0; i < 2; i++)
    x += i;

  assert(x == 99); // wrong: x is 4 here
  return 0;
}
