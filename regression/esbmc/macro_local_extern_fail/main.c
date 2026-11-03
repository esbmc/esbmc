/* A block-scope extern declared by a macro names the global it refers to, so
 * the pointer read through it is the global's. MatIEC binds located variables
 * this way. */
typedef struct
{
  unsigned char *value;
} slot;

#define BIND(location, s)                                                      \
  {                                                                            \
    extern unsigned char *location;                                            \
    (s).value = location;                                                      \
  }

static void init(slot *a, slot *b)
{
  BIND(p0, *a)
  BIND(p1, *b)
}

unsigned char i0, i1;
unsigned char *p0 = &i0, *p1 = &i1;

int main()
{
  slot a = {0}, b = {0};
  init(&a, &b);
  *a.value = 1;
  __ESBMC_assert(i0 == 1, "p0 bound");
  __ESBMC_assert(i1 == 1, "p1 written");
  return 0;
}
