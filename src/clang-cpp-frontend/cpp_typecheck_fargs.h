/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CLANG_CPP_FRONTEND_CLANG_CPP_FARGS_H_
#define CLANG_CPP_FRONTEND_CLANG_CPP_FARGS_H_

#include <util/std_code.h>

class cpp_typecheckt;
class cpp_typecast_rank;

class cpp_typecheck_fargst // for function overloading
{
public:
  bool in_use, has_object;
  exprt::operandst operands;

  // has_object indicates that the first element of
  // 'operands' is the 'this' pointer (with the object type,
  // not pointer to object type)

  cpp_typecheck_fargst() : in_use(false), has_object(false)
  {
  }

  bool has_class_type() const;

  void build(const side_effect_expr_function_callt &function_call);

  explicit cpp_typecheck_fargst(
    const side_effect_expr_function_callt &function_call)
    : in_use(false), has_object(false)
  {
    build(function_call);
  }

#if 0
  bool match(
    const code_typet &code_type,
    cpp_typecast_rank &distance,
    cpp_typecheckt &cpp_typecheck) const;
#endif

  void add_object(const exprt &expr)
  {
    //if(!in_use) return;
    has_object = true;
    operands.insert(operands.begin(), expr);
  }

  void remove_object()
  {
    assert(has_object);
    operands.erase(operands.begin());
    has_object = false;
  }
};

std::ostream &operator<<(std::ostream &out, const cpp_typecheck_fargst &fargs);

#endif /* CLANG_CPP_FRONTEND_CLANG_CPP_FARGS_H_ */
