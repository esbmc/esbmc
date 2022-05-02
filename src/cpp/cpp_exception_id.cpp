/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/cpp_exception_id.h>
#include <util/std_types.h>

#include <iostream>

void cpp_exception_list_rec(
  const typet &src,
  const namespacet &ns,
  const std::string &suffix,
  std::vector<irep_idt> &dest,
  bool is_catch)
{
  if(
    src.id() == "pointer" || src.id() == "array" ||
    src.id() == "incomplete_array")
  {
    if(src.reference())
    {
      // do not change
      cpp_exception_list_rec(src.subtype(), ns, suffix, dest, is_catch);
      return;
    }
    if(src.subtype().id() == "empty") // throwing void*
    {
      irep_idt identifier = "void_ptr";
      dest.emplace_back(id2string(identifier) + suffix);
    }
    else
    {
      // append suffix _ptr
      cpp_exception_list_rec(
        src.subtype(), ns, "_ptr" + suffix, dest, is_catch);
      return;
    }
  }
  else if(src.id() == "symbol")
  {
    irep_idt identifier = src.identifier();

    // We must check if is a derived class
    typet type = ns.lookup(identifier).type;

    if(
      type.id() == "struct" &&
      !is_catch) // We only get the base class when throwing
    {
      struct_typet struct_type = to_struct_type(type);
      const exprt &bases =
        static_cast<const exprt &>(struct_type.find("bases"));

      // Throwing a derived class
      if(bases.is_not_nil() && bases.get_sub().size())
      {
        // Save the derived class
        dest.emplace_back(id2string(identifier) + suffix);

        // Save all the base classes
        for(const auto &i : bases.get_sub())
        {
          typet base_type = i.type();
          identifier = base_type.identifier();
          dest.emplace_back(id2string(identifier) + suffix);
        }
      }
      else // Throwing a base class
        dest.emplace_back(id2string(identifier) + suffix);
    }
    else
      dest.emplace_back(id2string(identifier) + suffix);
  }
  else if(src.id() == "ellipsis")
  {
    irep_idt identifier = "ellipsis";
    dest.emplace_back(id2string(identifier) + suffix);
  }

  // grab C++ type
  irep_idt cpp_type = src.get("#cpp_type");
  if(cpp_type != irep_idt())
    dest.emplace_back(id2string(cpp_type) + suffix);
}

irept cpp_exception_list(const typet &src, const namespacet &ns)
{
  std::vector<irep_idt> ids;
  irept result("exception_list");

  cpp_exception_list_rec(src, ns, "", ids, false);
  result.get_sub().resize(ids.size());

  for(unsigned i = 0; i < ids.size(); i++)
    result.get_sub()[i].id(ids[i]);

  return result;
}

irep_idt cpp_exception_id(const typet &src, const namespacet &ns)
{
  std::vector<irep_idt> ids;
  cpp_exception_list_rec(src, ns, "", ids, true);
  assert(!ids.empty());
  return ids.front();
}
