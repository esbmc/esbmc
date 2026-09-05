/* Soundness twin: this table HAS null entries and the index is
 * nondet, so the miss path is genuinely reachable — the fold must
 * only fire for provably-nonnull addresses, never for a table read
 * that can yield null. */
typedef struct
{
  unsigned short cells[64];
  unsigned char sp;
  unsigned short pc;
} vm_t;
typedef int (*handler_t)(vm_t *);

static int op_push(vm_t *v)
{
  v->cells[v->sp] = 7;
  v->sp += 1;
  return 0;
}
static const handler_t table[4] = {op_push, 0, op_push, 0};
static vm_t vm;
unsigned int nondet_u4(void);

static int step(vm_t *v, unsigned int op)
{
  handler_t h = table[op & 3];
  if (h == 0)
    return 15;
  return h(v);
}

int main(void)
{
  vm.sp = 1;
  int st = step(&vm, nondet_u4());
  __ESBMC_assert(st == 0, "a null entry must keep its miss path");
  return 0;
}
