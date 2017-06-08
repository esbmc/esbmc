/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <util/c_qualifiers.h>
#include <cassert>
#include <cpp/cpp_typecheck.h>
#include <cpp/cpp_typecheck_fargs.h>
#include <util/std_types.h>

std::ostream &operator<<(std::ostream &out,
  const cpp_typecheck_fargst &fargs)
{
  out << "cpp_typecheck_fargst" << std::endl;
  out << "* in_use: " << fargs.in_use << std::endl;
  out << "* has_object: " << fargs.has_object << std::endl;

  if(fargs.operands.size())
  {
    out << "* operands: " << std::endl;
    for(unsigned i=0; i<fargs.operands.size(); ++i)
      out << "  " << i << ": " << fargs.operands[i] << std::endl;
  }

  return out;
}

/*******************************************************************\

Function: cpp_typecheck_fargst::has_class_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool cpp_typecheck_fargst::has_class_type() const
{
  for(exprt::operandst::const_iterator it=operands.begin();
      it!=operands.end();
      it++)
  {
    if(it->type().id()=="struct")
      return true;
  }

  return false;
}

/*******************************************************************\

Function: cpp_typecheck_fargst::build

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheck_fargst::build(
  const side_effect_expr_function_callt &function_call)
{
  in_use=true;

  operands.clear();
  operands.reserve(function_call.op1().operands().size());

  for(unsigned i=0; i<function_call.op1().operands().size(); i++)
    operands.push_back(function_call.op1().operands()[i]);
}

/*******************************************************************\

Function: cpp_typecheck_fargst::exact_match

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool cpp_typecheck_fargst::match(
  const code_typet &code_type,
  cpp_typecast_rank &distance,
  cpp_typecheckt &cpp_typecheck) const
{
  distance = cpp_typecast_rank();

  exprt::operandst ops = operands;
  const code_typet::argumentst &arguments=code_type.arguments();

  if(arguments.size()>ops.size())
  {
    // Check for default values.
    ops.reserve(arguments.size());

    for(unsigned i=ops.size(); i<arguments.size(); i++)
    {
      const exprt &default_value=
        arguments[i].default_value();

      if(default_value.is_nil())
        return false;

      ops.push_back(default_value);
    }
  }
  else if(arguments.size()<ops.size())
  {
    // check for ellipsis
    if(!code_type.has_ellipsis())
      return false;
  }

  for(unsigned i=0; i<ops.size(); i++)
  {
    // read
    // http://publib.boulder.ibm.com/infocenter/comphelp/v8v101/topic/com.ibm.xlcpp8a.doc/language/ref/implicit_conversion_sequences.htm
    //
    // The following are the three categories of conversion sequences in order from best to worst:
    // * Standard conversion sequences
    // * User-defined conversion sequences
    // * Ellipsis conversion sequences

    if(i>=arguments.size())
    {
      // Ellipsis is the 'worst' of the conversion sequences
      distance.rank+=1000;
      continue;
    }

    exprt argument=arguments[i];

    exprt &operand=ops[i];

    #if 0
    // unclear, todo
    if(is_reference(operand.type()))
      std::cout << "O: " << operand.pretty() << std::endl;

    assert(!is_reference(operand.type()));
    #endif

    // "this" is a special case -- we turn the pointer type
    // into a reference type to do the type matching
    if(i==0 && argument.cmt_base_name()=="this")
    {
      argument.type().set("#reference", true);
      argument.type().set("#this", true);
    }

    cpp_typecast_rank rank;
    exprt new_expr;

    // can we do the standard conversion sequence?
    if(cpp_typecheck.implicit_conversion_sequence(
        operand, argument.type(), new_expr, rank))
    {
      operand.swap(new_expr);

      // ok
      distance+=rank;
    }
    else
    {
      return false; // no conversion possible
    }
  }

  return true;
}
