/*******************************************************************\

Module: SMT-LIB Builtin Logics 

Author: CM Wintersteiger

\*******************************************************************/

#ifndef SMT_LOGICS_H_
#define SMT_LOGICS_H_

#include "smt_finalizer.h"

smt_finalizert* create_smt_finalizer(
  const std::string &,
  contextt &,
  message_handlert &);

#endif /*SMT_LOGICS_H_*/
