/* #10008: printing this initialiser read past its operands. */
int main(void)
{
  union { struct { int s; int c; }; unsigned i; } v = { .s = 17 };
  return v.i;
}
