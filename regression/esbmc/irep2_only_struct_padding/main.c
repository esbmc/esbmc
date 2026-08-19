/* Padding is part of the layout, not a spelling: the symbol table's type is
   what ESBMC sizes objects and computes member offsets from. */
struct s
{
  int a;
  char c;
};

struct t
{
  char c;
  int a;
};

extern struct s g;
extern struct t h;

int main(void)
{
  return g.a + h.a;
}
