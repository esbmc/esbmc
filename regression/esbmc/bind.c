#include "bind.h"

int dn_expand(const u_char *msg, const u_char *eomorig,
              const u_char *comp_dn, char *exp_dn, int length)
{
  if (nondet_int ())
    return -1;
  else {
    exp_dn[length-1] = (u_char) EOS;
  }
  return strlen(comp_dn);
}
