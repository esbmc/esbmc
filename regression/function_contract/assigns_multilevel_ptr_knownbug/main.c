/*
 * assigns_multilevel_ptr_knownbug:
 *   __ESBMC_assigns(o->sub->a) declares only the nested field 'a' as writable.
 *   The body also writes o->x which is NOT in the assigns clause — a violation.
 *   ESBMC's Phase 2C only handles single-level pointer dereferences (*ptr or
 *   ptr->field); it does not classify multi-level patterns like ptr->sub->field.
 *   The assigns compliance check generates 0 VCCs for this case and reports
 *   VERIFICATION SUCCESSFUL (false negative / unsound).
 *
 *   Known limitation: multi-level pointer assigns (ptr->sub->field) are unsupported.
 *
 *   Expected (correct): VERIFICATION FAILED — assigns violation on o->x
 *   Current (bug):      VERIFICATION SUCCESSFUL — false negative
 */
typedef struct
{
  int a;
  int b;
} Inner;

typedef struct
{
  Inner *sub;
  int x;
} Outer;

void write_sub_a(Outer *o, int v)
{
  __ESBMC_requires(o != (void *)0 && o->sub != (void *)0);
  __ESBMC_assigns(o->sub->a);
  __ESBMC_ensures(1);
  o->sub->a = v;
  o->x = 99; /* assigns violation: o->x is NOT in __ESBMC_assigns */
}

int main()
{
  return 0;
}
