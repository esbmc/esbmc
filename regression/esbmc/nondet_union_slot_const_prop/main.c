/* The tagged-slot VM idiom: an operand stack is an array of unions.
 * One nondet stored into a slot member (dead: overwritten on the next
 * line) must not stop the rest of the machine state from constant-
 * propagating — before the fix it made the whole vm literal opaque:
 * the dispatch loop's exit never folded, and symex unrolled the
 * data-independent handler-table walk to the unwind bound. At
 * --unwind 20000 this verifies in constant time when the fold works
 * and cannot complete when it does not. */
typedef unsigned short u2;
typedef union
{
  short s;
  u2 r;
} slot_t;
u2 nondet_u2(void);

struct entry
{
  u2 start, end, handler;
};
struct m
{
  const struct entry *table;
  u2 n;
};
struct fr
{
  struct m *method;
  u2 pc;
  unsigned char sp;
  slot_t stack[16];
};
struct vm
{
  struct fr frame;
  u2 depth;
};
static struct entry et[1];
static struct m mi;
static struct vm vm;

static u2 find(const struct m *method, u2 pc)
{
  for (u2 i = 0; i < method->n; i++)
    if (pc >= method->table[i].start && pc < method->table[i].end)
      return method->table[i].handler;
  return (u2)-1;
}

static int dispatch(struct vm *v)
{
  while (1)
  {
    u2 h = find(v->frame.method, v->frame.pc);
    if (h != (u2)-1)
    {
      v->frame.pc = h;
      return 0;
    }
    if (v->depth == 0)
      return 1;
    v->depth--;
  }
}

int main(void)
{
  et[0].start = 0;
  et[0].end = 1;
  et[0].handler = 7;
  mi.table = et;
  mi.n = 1;
  vm.frame.method = &mi;
  vm.frame.pc = 0;
  vm.frame.stack[0].r = nondet_u2(); /* dead: overwritten next */
  vm.frame.stack[0].r = 1;
  vm.frame.sp = 1;
  __ESBMC_assert(dispatch(&vm) == 0, "caught");
  __ESBMC_assert(vm.frame.pc == 7, "at the handler");
  return 0;
}
