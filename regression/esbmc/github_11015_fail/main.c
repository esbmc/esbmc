/* #11015: a whole struct stored through a misaligned pointer. */
struct u2 { unsigned short h; };
struct d { unsigned long long x; };
int main(void)
{
  struct d d = {0};
  struct u2 w = {0xffee};
  *(struct u2 *)((unsigned char *)&d + 3) = w;
  return (int)(d.x >> 24);
}
