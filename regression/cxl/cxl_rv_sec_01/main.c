// CXL PMEM security state machine as a Linux RV automaton.
//
// cxl_sec.dot encodes the rules in drivers/cxl/security.c against CXL 2.0
// §8.2.9.8.6: a passphrase may be set or disabled while unlocked, a locked
// device must be unlocked before use, and freeze is terminal -- once frozen,
// no passphrase operation is accepted at all.
// Expected: VERIFICATION SUCCESSFUL

#include <assert.h>
#include "rv_cxl_sec.h"

extern unsigned int __VERIFIER_nondet_uint(void);
void __ESBMC_assume(_Bool);

int main(void)
{
  rv_cxl_sec_reset();

  rv_cxl_sec_event(set_passphrase);
  rv_cxl_sec_event(set_passphrase);   /* re-keying while unlocked is allowed */
  rv_cxl_sec_event(lock);
  rv_cxl_sec_event(unlock);

  /* A driver may give up on security entirely... */
  rv_cxl_sec_event(disable_passphrase);
  assert(rv_cxl_sec_cur == sec_disabled);

  /* ...or freeze, after which nothing further is legal. The automaton has no
     outgoing edge from sec_frozen, so any later event would be rejected. */
  rv_cxl_sec_event(freeze);
  assert(rv_cxl_sec_cur == sec_frozen);

  return 0;
}
