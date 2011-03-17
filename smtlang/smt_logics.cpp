/*******************************************************************\

Module: SMT-LIB Builtin Logics 

Author: CM Wintersteiger

\*******************************************************************/

#include "smt_logics.h"
#include "smt_typecheck.h"

#include "smt_finalizer.h"
#include "smt_finalizer_generic.h"
#include "smt_finalizer_AUFLIA.h"
#include "smt_finalizer_AUFLIRA.h"
#include "smt_finalizer_AUFNIRA.h"
#include "smt_finalizer_QF_IDL.h"
#include "smt_finalizer_QF_LIA.h"
#include "smt_finalizer_QF_LRA.h"
#include "smt_finalizer_QF_UFBV32.h"

/*******************************************************************\

Function: get_smt_finalizer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

smt_finalizert* create_smt_finalizer(
  const std::string &logic,
  contextt &context,
  message_handlert &message_handler)
{
  symbolst::const_iterator it = 
    context.symbols.find(smt_typecheckt::lbase+logic);
  if (it!=context.symbols.end() && !it->second.is_extern)
  {
    // logic file has been loaded - use the generic checker.
    return new smt_finalizer_generict(context, message_handler);
  }
    
  if (logic=="AUFLIA")
    return new smt_finalizer_AUFLIAt(context, message_handler);
  if (logic=="AUFLIRA")
    return new smt_finalizer_AUFLIRAt(context, message_handler);
  if (logic=="AUFNIRA")
    return new smt_finalizer_AUFNIRAt(context, message_handler);
  if (logic=="QF_IDL")
    return new smt_finalizer_QF_IDLt(context, message_handler);
  if (logic=="QF_LIA")
    return new smt_finalizer_QF_LIAt(context, message_handler);
  if (logic=="QF_LRA")
    return new smt_finalizer_QF_LRAt(context, message_handler);
  if (logic=="QF_UFBV[32]")
    return new smt_finalizer_QF_UFBV32t(context, message_handler);
     
  return new smt_finalizer_generict(context, message_handler);
}
