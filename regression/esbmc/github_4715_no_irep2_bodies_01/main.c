// V.4.4 (esbmc/esbmc#4715): --irep2-bodies is now the default goto-convert
// path. This test pins the --no-irep2-bodies escape hatch: it must still parse,
// select the legacy body-lowering path, and produce the correct verdict over a
// mix of structured control flow (if / while / for / switch).

int main()
{
  int x = 0;

  if (1)
    x = 1;

  while (x < 3)
    x++;

  for (int i = 0; i < 2; i++)
    x += i;

  int z = 0;
  switch (x)
  {
  case 4:
    z = 1;
    break;
  default:
    z = 2;
  }

  assert(x == 4);
  assert(z == 1);
  return 0;
}
