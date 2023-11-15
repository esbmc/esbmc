typedef unsigned int    size_t;
typedef int             ssize_t;
typedef int             atomic_t;
typedef unsigned        gfp_t;

#define atomic_set(v,i)	(*v) = i

struct pp_struct {
  atomic_t irqc;
};

void * kmalloc(size_t size, gfp_t flags)
{
  return malloc(size);
}

int main(void)
{
  struct pp_struct *pp;

  pp = kmalloc (sizeof (struct pp_struct), 10);

  // This works:
  // pp = malloc (sizeof (struct pp_struct));

  if (!pp)
    return -10;
  
  atomic_set (&pp->irqc, 0);
  
  return 0;
}
