/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/cpp_typecheck.h>
#include <util/i2string.h>

void cpp_typecheckt::typecheck_function_bodies()
{
  instantiation_stackt old_instantiation_stack;
  old_instantiation_stack.swap(instantiation_stack);

  while(!function_bodies.empty())
  {
    symbolt &function_symbol = *function_bodies.front().function_symbol;
    template_map.swap(function_bodies.front().template_map);
    instantiation_stack.swap(function_bodies.front().instantiation_stack);

    if(function_symbol.id.as_string() == "Bird::Bird(this)")
      printf("Got Bird ctor\n");
    if(function_symbol.id.as_string() == "Bird::~Bird(this)")
      printf("Got Bird dtor\n");
    if(function_symbol.id.as_string() == "Penguin::Penguin(this)")
      printf("Got Penguin ctor\n");
    if(function_symbol.id.as_string() == "Penguin::~Penguin(this)")
      printf("Got Penguin dtor\n");

    function_bodies.pop_front();

    if(function_symbol.id == "main")
      add_argc_argv(function_symbol);

    exprt &body = function_symbol.value;
    if(body.id() == "cpp_not_typechecked")
      continue;

    if(body.is_not_nil() && !body.is_zero())
    {
      convert_function(function_symbol);
    }

    printf("done\n");
  }

  old_instantiation_stack.swap(instantiation_stack);
}
