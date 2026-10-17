/* github #8659 */
int nondet_int();
int main(void)
{
  int x = nondet_int();
  int r = 0 - x;
  return r;
}
