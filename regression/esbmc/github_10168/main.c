/* esbmc/esbmc#10168: a switch with no case still evaluates its value. */
int main(void)
{
  int *p = (int *)0;
  switch (*p)
  {
  }
  return 0;
}
