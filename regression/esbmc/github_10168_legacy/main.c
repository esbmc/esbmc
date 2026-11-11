/* esbmc/esbmc#10168, through the legacy goto_convertt::convert_switch. */
int main(void)
{
  int *p = (int *)0;
  switch (*p)
  {
  default:
    break;
  }
  return 0;
}
