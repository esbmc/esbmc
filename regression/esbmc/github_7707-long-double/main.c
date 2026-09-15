/* ESBMC models `long double` as 12 bytes under --32, and 12 is not an alignment
 * any object can carry -- the offset predicate's mask is nonsense there. The
 * base check is confined to power-of-two widths for that reason: without the
 * guard this reports misaligned against a requirement no address satisfies. */

struct __attribute__((packed)) S
{
  long double b;
  char a;
};

int main(void)
{
  struct S sp;
  long double *p = (long double *)&sp.b;
  long double z = *p;
  (void)z;
  return 0;
}
