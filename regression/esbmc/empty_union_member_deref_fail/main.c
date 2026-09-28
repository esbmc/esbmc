/* The byte at offset sizeof(struct s) is past the end: the empty union member
 * contributes no storage, so the access must be reported out of bounds rather
 * than aborting the zero-size skip (#5393). */
union e
{
};

struct s
{
  union e u;
  long a;
};

int main(void)
{
  struct s t;
  t.a = 0;
  char *c = (char *)&t;
  c[sizeof(struct s)] = 1;
  return 0;
}
