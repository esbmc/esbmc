// Fixture for scripts/complexity/test_ccn_report.py. Not built.
// The `before`/`after` pair stands in for a PR's merge base and head.

int worsened(int a)
{
  int r = 0;
  if (a == 0)
    r++;
  if (a == 1)
    r++;
  if (a == 2)
    r++;
  if (a == 3)
    r++;
  if (a == 4)
    r++;
  if (a == 5)
    r++;
  if (a == 6)
    r++;
  if (a == 7)
    r++;
  if (a == 8)
    r++;
  if (a == 9)
    r++;
  if (a == 10)
    r++;
  return r;
}

int grew_a_little(int a)
{
  int r = 0;
  if (a == 0)
    r++;
  if (a == 1)
    r++;
  return r;
}

int tiny(int a)
{
  int r = 0;
  if (a == 0)
    r++;
  return r;
}
