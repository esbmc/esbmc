/* A long double read through a pointer to a packed struct's member is checked
 * against its full width. Under --32 that width used to be 12 bytes, which is
 * not a power of two, so the base check was skipped; it is 16 bytes now
 * (#8097), the same as under --64, so the misaligned read is reported. */

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
