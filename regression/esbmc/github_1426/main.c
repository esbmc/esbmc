// based on esbmc/github_732-2-2

typedef union {
  int x[12000];
} t1;

typedef struct {
  int x;
  t1 y;
} t2;

t2 nondet_t2();

int main()
{
  t2 a = nondet_t2();
  int *b = &a.x;
  b[1] = a.x;
  assert(*b);
}
