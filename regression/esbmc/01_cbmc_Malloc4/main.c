int nondet_int();
void *malloc(unsigned s);
void free(void *p);

typedef struct {
    int i;
} s;

typedef s *s_t;

int main()
{
  s_t p;

  p = (s_t)malloc(sizeof(s_t));
  free(p);
  // this should fail, as the object is not alive anymore
  p->i = nondet_int();
}
