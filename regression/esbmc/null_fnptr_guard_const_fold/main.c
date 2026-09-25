/* Distilled from verifying a JavaCard VM's interpreter loop: handler
 * dispatch through a static function-pointer table, one level deeper
 * than the naive shape — the null check and the handler call live in a
 * callee, and the handler mutates a shared machine struct. Without the fold the impossible
 * null path survives to the callee's return merge, the whole struct
 * becomes a phi of updated-vs-original, and every later member read
 * -- including the loop bound -- stops folding. */
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
  v->pc += 1;
  return 0;
}
static const handler_t table[4] = {op_push, op_push, op_push, op_push};
static vm_t vm;

static int step(vm_t *v)
{
  handler_t h = table[v->pc & 3];
  if (h == 0)
    return 15;
  return h(v);
}

int main(void)
{
  vm.sp = 1;
  int st = step(&vm);
  int n = 0;
  for (unsigned short i = 0; i < (st == 0 ? vm.sp : 30000); i++)
    n++;
  __ESBMC_assert(n == 2, "the null guard folded; sp bounded the loop");

  /* The other polarity: a `!= NULL` guard over a provably real address
   * must fold true the same way. */
  int m = 0;
  if (table[(vm.pc + 1) & 3] != 0)
    for (unsigned short i = 0; i < vm.sp; i++)
      m++;
  __ESBMC_assert(m == 2, "the nonnull guard folded too");

  /* Constant-offset forms: an object base plus or minus a constant is
   * still a real object's address in the (object, offset) model, even
   * past the object's end. */
  unsigned short *q = vm.cells + 8;
  int k = 0;
  if (q != 0 && q - 3 != 0 && 0 != q + 60)
    for (unsigned short i = 0; i < vm.sp; i++)
      k++;
  __ESBMC_assert(k == 2, "the offset guards folded too");
  return 0;
}
