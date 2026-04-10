typedef struct {
  long a;
} b;

struct {
  char c;
} typedef d;

typedef struct {
  struct {
    char : 1;
  };
} e;

typedef struct {
  b f;
  e attributes;
} g;

g h[];
d *i;
g *j, *k;

void main() {
  int l;
  k = h;
  j = &k[l];
  g *a = j;

  switch (a->f.a)
  case 0:
    i->c;
}
