/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <location.h>

#include "cpp_typecheck.h"

/*******************************************************************\

Function: cpp_typecheckt::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::convert(cpp_usingt &cpp_using)
{
  // there are two forms of using clauses:
  // a) using namespace SCOPE;  ("using directive")
  // b) using SCOPE::id;        ("using declaration")

  cpp_typecheck_resolvet resolver(*this);
  cpp_save_scopet save_scope(this->cpp_scopes);

  std::string base_name;
  cpp_template_args_non_tct template_args;
  resolver.resolve_scope(cpp_using.name(), base_name, template_args);

  bool qualified=cpp_using.name().is_qualified();
  cpp_scopest::id_sett id_set;

  this->cpp_scopes.get_ids(base_name, id_set, qualified);

  bool using_directive=cpp_using.get_namespace();

  if(id_set.empty())
  {
    err_location(cpp_using.name().location());
    str << "using "
        << (using_directive?"namespace":"identifier")
        << " `"
        << base_name << "' not found";
    throw 0;
  }

  // go back to where we used to be
  save_scope.restore();

  for(cpp_scopest::id_sett::iterator
      it=id_set.begin();
      it!=id_set.end();
      it++)
  {
    cpp_scopes.current_scope().using_set.insert(*it);
  }
}
