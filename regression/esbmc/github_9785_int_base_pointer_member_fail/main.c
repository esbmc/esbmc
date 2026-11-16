/* #9785: a pointer stored into a struct through an integer address and read
 * back keeps its object; its bytes used to tie it to every other pointer
 * flattened to bytes, pairwise. */
extern int __VERIFIER_nondet_int(void);
struct port
{
  unsigned long f[4];
  unsigned char *mb;
};
static struct port ports[1];
static unsigned char io[8];
int main(void)
{
  int i = __VERIFIER_nondet_int();
  if (i < 0 || i >= 1)
    return 0;
  unsigned long base = (unsigned long)ports + i * sizeof(struct port);
  *((unsigned long *)(base + 0)) = 1UL;
  *((unsigned long *)(base + 8)) = 2UL;
  *((unsigned long *)(base + 16)) = 3UL;
  *((unsigned long *)(base + 24)) = 4UL;
  *((unsigned char **)(base + 32)) = io;
  unsigned char *m = *((unsigned char **)(base + 32));
  __ESBMC_assert(m != io, "the stored pointer is read back");
  return 0;
}
