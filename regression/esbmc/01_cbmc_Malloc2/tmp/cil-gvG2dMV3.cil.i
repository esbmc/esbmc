# 1 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/tmp/cil-dsjZbXXf.cil.c"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/tmp/cil-dsjZbXXf.cil.c"
# 1 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/main.c"
typedef unsigned int size_t;
# 3 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/main.c"
typedef int atomic_t;
# 4 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/main.c"
typedef unsigned int gfp_t;
# 8 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/main.c"
struct pp_struct {
   atomic_t irqc ;
};
# 12 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/main.c"
extern void *malloc(size_t size ) ;
# 14 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/main.c"
void *kmalloc(size_t size , gfp_t flags )
{ void *tmp ;

  {
# 16 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/main.c"
  tmp = malloc(size);
# 16 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/main.c"
  return (tmp);
}
}
# 19 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/main.c"
int main(void)
{ struct pp_struct *pp ;
  void *tmp ;

  {
# 23 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/main.c"
  tmp = kmalloc(sizeof(struct pp_struct ), 10U);
# 23 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/main.c"
  pp = (struct pp_struct *)tmp;
# 28 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/main.c"
  if (! pp) {
# 29 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/main.c"
    return (-10);
  }
# 31 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/main.c"
  pp->irqc = 0;
# 33 "/home/lucas/cprover/regression/regression/esbmc/01_cbmc_Malloc2/main.c"
  return (0);
}
}
