#include <global.h>
#include <stdio.h>
#include <string>
#include <global.h>
#include <cstdarg>

#include "z3_conv.h"
#include "z3_capi.h"

void
abortf(const char* message)
{
  fprintf(stderr, "%s\n", message);
  abort();
}

Z3_lbool
z3_capi::check2(Z3_lbool expected_result)
{
  Z3_model m      = 0;
  Z3_lbool result = Z3_check_and_get_model(z3_ctx, &m);
  switch (result) {
  case Z3_L_FALSE:
    break;

  case Z3_L_UNDEF:
    break;

  case Z3_L_TRUE:
    break;
  }
  if (m) {
    Z3_del_model(z3_ctx, m);
  }
  if (result != expected_result) {
    abortf("unexpected result");
  }

  return result;
}
