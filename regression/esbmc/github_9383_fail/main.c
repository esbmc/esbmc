/* #9383: one byte past an array whose element has an unnamed bit-field. */
struct A
{
  char c;
  int : 4;
};
struct A a[3];

int main(void)
{
  char *b = (char *)a;
  b[sizeof a + 1] = 1;
  return 0;
}
