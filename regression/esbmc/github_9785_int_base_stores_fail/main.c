/* #9785: stores and loads through an integer address
 * ((unsigned long)ports + i * size + offset) made the solver search the address
 * space for each; these 16 fields took 82 s, and a pointer member far more. */
extern int __VERIFIER_nondet_int(void);
struct port
{
  unsigned long f[16];
};
static struct port ports[1];
int main(void)
{
  int i = __VERIFIER_nondet_int();
  if (i < 0 || i >= 1)
    return 0;
  unsigned long base = (unsigned long)ports + i * sizeof(struct port);
  *((unsigned long *)(base + 0)) = 0UL;
  *((unsigned long *)(base + 8)) = 1UL;
  *((unsigned long *)(base + 16)) = 2UL;
  *((unsigned long *)(base + 24)) = 3UL;
  *((unsigned long *)(base + 32)) = 4UL;
  *((unsigned long *)(base + 40)) = 5UL;
  *((unsigned long *)(base + 48)) = 6UL;
  *((unsigned long *)(base + 56)) = 7UL;
  *((unsigned long *)(base + 64)) = 8UL;
  *((unsigned long *)(base + 72)) = 9UL;
  *((unsigned long *)(base + 80)) = 10UL;
  *((unsigned long *)(base + 88)) = 11UL;
  *((unsigned long *)(base + 96)) = 12UL;
  *((unsigned long *)(base + 104)) = 13UL;
  *((unsigned long *)(base + 112)) = 14UL;
  *((unsigned long *)(base + 120)) = 15UL;
  unsigned long s = 0;
  s += *((unsigned long *)(base + 0));
  s += *((unsigned long *)(base + 8));
  s += *((unsigned long *)(base + 16));
  s += *((unsigned long *)(base + 24));
  s += *((unsigned long *)(base + 32));
  s += *((unsigned long *)(base + 40));
  s += *((unsigned long *)(base + 48));
  s += *((unsigned long *)(base + 56));
  s += *((unsigned long *)(base + 64));
  s += *((unsigned long *)(base + 72));
  s += *((unsigned long *)(base + 80));
  s += *((unsigned long *)(base + 88));
  s += *((unsigned long *)(base + 96));
  s += *((unsigned long *)(base + 104));
  s += *((unsigned long *)(base + 112));
  s += *((unsigned long *)(base + 120));
  __ESBMC_assert(s == 121UL, "sum of the stored fields");
  return 0;
}
