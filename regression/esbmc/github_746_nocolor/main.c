int main(void)
{
  void *p = &&lbl;   /* indirect goto: unsupported, triggers an AST dump */
lbl:
  goto *p;
  return 0;
}
