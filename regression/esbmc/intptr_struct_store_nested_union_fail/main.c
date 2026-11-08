union lock
{
  unsigned int word;
  struct
  {
    unsigned short head, tail;
  } half;
};
struct counts
{
  unsigned int rx, tx;
};
struct port
{
  union lock lock;
  struct counts counts;
  unsigned long base;
};
static struct port ports[2];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  unsigned long p = (unsigned long)ports + (unsigned long)i * sizeof(struct port);
  *(unsigned short *)(p + __builtin_offsetof(struct port, lock.half.tail)) = 5;
  *(unsigned int *)(p + __builtin_offsetof(struct port, counts.tx)) = 7;
  __ESBMC_assert(ports[i].lock.half.tail == 5, "tail stored");
  __ESBMC_assert(ports[i].lock.half.head == 0, "head untouched");
  __ESBMC_assert(ports[i].counts.tx == 7, "tx stored");
  __ESBMC_assert(ports[i].counts.rx == 0, "rx untouched");
  __ESBMC_assert(ports[i].lock.half.head == 5, "head written");
  return 0;
}
