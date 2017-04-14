/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/cpp_exception_id.h>
#include <util/std_types.h>

/*******************************************************************\

Function: cpp_exception_list_rec

  Inputs:

 Outputs:

 Purpose: turns a type into a list of relevant exception IDs

\*******************************************************************/

#include <iostream>

void cpp_exception_list_rec(
  const typet &src,
  const namespacet &ns,
  const std::string &suffix,
  std::vector<irep_idt> &dest,
  bool is_catch)
{
  if(src.id()=="pointer"
    || src.id()=="array"
    || src.id()=="incomplete_array")
  {
    if(src.reference())
    {
      // do not change
      cpp_exception_list_rec(src.subtype(), ns, suffix, dest, is_catch);
      return;
    }
    else if(src.subtype().id()=="empty") // throwing void*
    {
      irep_idt identifier = "void_ptr";
      dest.push_back(id2string(identifier)+suffix);
    }
    else
    {
      // append suffix _ptr
      cpp_exception_list_rec(src.subtype(), ns, "_ptr"+suffix, dest, is_catch);
      return;
    }
  }
  else if(src.id()=="symbol")
  {
    irep_idt identifier = src.identifier();

    // We must check if is a derived class
    typet type = ns.lookup(identifier).type;

    if(type.id()=="struct"
       && !is_catch) // We only get the base class when throwing
    {
      struct_typet struct_type=to_struct_type(type);
      const exprt &bases = static_cast<const exprt&>(struct_type.find("bases"));

      // Throwing a derived class
      if(bases.is_not_nil()
         && bases.get_sub().size())
      {
        // Save the derived class
        dest.push_back(id2string(identifier)+suffix);

        // Save all the base classes
        for(unsigned int i=0; i<bases.get_sub().size(); ++i)
        {
          typet base_type = bases.get_sub()[i].type();
          identifier = base_type.identifier();
          dest.push_back(id2string(identifier)+suffix);
        }
      }
      else // Throwing a base class
        dest.push_back(id2string(identifier)+suffix);
    }
    else
      dest.push_back(id2string(identifier)+suffix);
  }
  else if(src.id()=="ellipsis")
  {
    irep_idt identifier = "ellipsis";
    dest.push_back(id2string(identifier)+suffix);
  }

  // grab C++ type
  irep_idt cpp_type=src.get("#cpp_type");

  if(cpp_type!=irep_idt())
  {
    dest.push_back(id2string(cpp_type)+suffix);
    return;
  }

  return;
}

/*******************************************************************\

Function: cpp_exception_list

  Inputs:

 Outputs:

 Purpose: turns a type into a list of relevant exception IDs

\*******************************************************************/

irept cpp_exception_list(const typet &src, const namespacet &ns)
{
  std::vector<irep_idt> ids;
  irept result("exception_list");

  cpp_exception_list_rec(src, ns, "", ids, false);
  result.get_sub().resize(ids.size());

  for(unsigned i=0; i<ids.size(); i++)
    result.get_sub()[i].id(ids[i]);

  return result;
}

/*******************************************************************\

Function: cpp_exception_id

  Inputs:

 Outputs:

 Purpose: turns a type into an exception ID

\*******************************************************************/

irep_idt cpp_exception_id(const typet &src, const namespacet &ns)
{
  std::vector<irep_idt> ids;
  cpp_exception_list_rec(src, ns, "", ids, true);
  assert(!ids.empty());
  return ids.front();
}
