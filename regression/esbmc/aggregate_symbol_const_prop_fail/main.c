/* Mirror of the JCVM frame idiom: union-cell stack + type-tag array inside
 * a struct accessed through a pointer, shuffle at concrete indices. */
typedef union { short s; unsigned short r; } slot_t;
typedef struct {
  const unsigned char *code;
  unsigned short code_length;
  unsigned short pc;
  slot_t stack[256];
  unsigned char stack_types[256];
  unsigned short sp;
} frame_t;
typedef struct { frame_t frame; } vm_t;

short nondet_short(void);
unsigned char nondet_uchar(void);

static vm_t vm;

static int step(vm_t *v) {
  frame_t *f = &v->frame;
  unsigned char m = 1, n = 5;
  slot_t saved[4];
  unsigned char saved_types[4];
  for (unsigned char i = 0; i < m; i++) {
    saved[i] = f->stack[f->sp - m + i];
    saved_types[i] = f->stack_types[f->sp - m + i];
  }
  for (signed char i = (signed char)n - 1; i >= 0; i--) {
    f->stack[f->sp - n + m + i] = f->stack[f->sp - n + i];
    f->stack_types[f->sp - n + m + i] = f->stack_types[f->sp - n + i];
  }
  for (unsigned char i = 0; i < m; i++) {
    f->stack[f->sp - n + i] = saved[i];
    f->stack_types[f->sp - n + i] = saved_types[i];
  }
  f->sp += m;
  return 0;
}

int main(void) {
  vm.frame.sp = 5;
  short v0 = nondet_short(); unsigned char t0 = nondet_uchar();
  vm.frame.stack[0].s = v0; vm.frame.stack_types[0] = t0;
  short v1 = nondet_short(); unsigned char t1 = nondet_uchar();
  vm.frame.stack[1].s = v1; vm.frame.stack_types[1] = t1;
  short v2 = nondet_short(); unsigned char t2 = nondet_uchar();
  vm.frame.stack[2].s = v2; vm.frame.stack_types[2] = t2;
  short v3 = nondet_short(); unsigned char t3 = nondet_uchar();
  vm.frame.stack[3].s = v3; vm.frame.stack_types[3] = t3;
  short v4 = nondet_short(); unsigned char t4 = nondet_uchar();
  vm.frame.stack[4].s = v4; vm.frame.stack_types[4] = t4;
  step(&vm);
  __ESBMC_assert(vm.frame.sp == 6, "sp");
  __ESBMC_assert(vm.frame.stack[0].s == v4, "dup landed");
  __ESBMC_assert(vm.frame.stack[5].s == v4, "top kept");
  __ESBMC_assert(vm.frame.stack[1].s == v1, "wrong shuffle oracle");
  return 0;
}
