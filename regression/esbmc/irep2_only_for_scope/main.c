/* The loop variable's scope is the loop, so its DEAD belongs after the body
   that reads it. clang_c_adjust gets that by hoisting the init into an
   enclosing block; the hoist must splice a block-shaped init rather than
   nesting it, or the scope closes immediately. */
int main(void)
{
  int s = 0;
  for (int i = 0; i < 3; i++)
    s += i;
  return s;
}
