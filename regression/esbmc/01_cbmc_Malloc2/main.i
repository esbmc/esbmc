# 1 "main.c"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "main.c"
typedef unsigned int size_t;
typedef int ssize_t;
typedef int atomic_t;
typedef unsigned gfp_t;



struct pp_struct {
  atomic_t irqc;
};

void *malloc(size_t size);

void * kmalloc(size_t size, gfp_t flags)
{
  return malloc(size);
}

int main(void)
{
  struct pp_struct *pp;

  pp = kmalloc (sizeof (struct pp_struct), 10);




  if (!pp)
    return -10;

  (*&pp->irqc) = 0;

  return 0;
}
