#include <assert.h>

/* The guard protects both callers of declare_gcc_polymorphic_builtin. The
 * prefix_no_args/_nonptr pair pins the legacy one; this pins the IREP2 one,
 * which reaches it through clang_c_adjust_irep2::declare_polymorphic_builtin. */
int __sync_fetch_and_add_mine(void);
int __atomic_load_n_bogus(int);

int main(void)
{
  int a = __sync_fetch_and_add_mine();
  int b = __atomic_load_n_bogus(3);
  assert(a == a && b == b);
  return 0;
}
