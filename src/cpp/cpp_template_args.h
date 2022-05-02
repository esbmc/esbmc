/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_TEMPLATE_ARGS_H
#define CPROVER_CPP_TEMPLATE_ARGS_H

#include <util/std_expr.h>

// A data structures for template arguments, i.e.,
// expressions of the form <E1, T2, ...>.
// Not to be confused with the template parameters!

class cpp_template_args_baset : public irept
{
public:
  cpp_template_args_baset() : irept("template_args")
  {
  }

  typedef std::vector<exprt> argumentst;

  argumentst &arguments()
  {
    return (argumentst &)(add("arguments").get_sub());
  }

  const argumentst &arguments() const
  {
    return (const argumentst &)(find("arguments").get_sub());
  }
};

// the non-yet typechecked variant

class cpp_template_args_non_tct : public cpp_template_args_baset
{
};

extern inline cpp_template_args_non_tct &
to_cpp_template_args_non_tc(irept &irep)
{
  assert(irep.id() == "template_args");
  return static_cast<cpp_template_args_non_tct &>(irep);
}

extern inline const cpp_template_args_non_tct &
to_cpp_template_args_non_tc(const irept &irep)
{
  assert(irep.id() == "template_args");
  return static_cast<const cpp_template_args_non_tct &>(irep);
}

// the already typechecked variant

class cpp_template_args_tct : public cpp_template_args_baset
{
public:
  bool has_unassigned() const
  {
    const argumentst &_arguments = arguments();
    for(const auto &_argument : _arguments)
      if(
        _argument.id() == "unassigned" || _argument.type().id() == "unassigned")
        return true;

    return false;
  }
};

extern inline cpp_template_args_tct &to_cpp_template_args_tc(irept &irep)
{
  assert(irep.id() == "template_args");
  return static_cast<cpp_template_args_tct &>(irep);
}

extern inline const cpp_template_args_tct &
to_cpp_template_args_tc(const irept &irep)
{
  assert(irep.id() == "template_args");
  return static_cast<const cpp_template_args_tct &>(irep);
}

#endif
