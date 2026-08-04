/* Pins that the clang AST dump the frontend emits for an unsupported
   construct honours --color never. The construct is incidental -- it was an
   indirect goto until that became supported (issue #4083). */
_Atomic int a;

int main(void)
{
  __c11_atomic_fetch_nand(&a, 1, 0);
  return 0;
}
