/* #5393: a flexible array member adds nothing to the size of its structure
 * (C17 6.7.2.1p18), so the byte at offset sizeof(struct s) lies past the end
 * of a `struct s` local. Sizing the member as a one-element array hides this
 * overflow. */
struct s
{
  void *a;
  long slots[];
};

int main(void)
{
  struct s t;
  t.a = 0;
  char *c = (char *)&t;
  c[sizeof(struct s)] = 1;
  return 0;
}
