/* #10090: a store that starts in an array member at an odd offset and ends in
 * the next member is still misaligned. */
struct S
{
  unsigned char c[6];
  unsigned short h;
};
struct S s[2];
int main(void)
{
  unsigned char *p = (unsigned char *)&s[1] + 5;
  *(unsigned short *)p = 0x0201u;
  return 0;
}
