#include <clang-cpp-frontend/clang_cpp_exception_id.h>

/// A class type's own id followed by its bases', most derived first, so a
/// handler for a base catches a derived throw. Only a throw expands the bases:
/// `catch (Base)` names one id and matches by the list the throw carries.
static void class_exception_ids(
  const namespacet &ns,
  const typet &type,
  const std::string &suffix,
  std::vector<irep_idt> &ids,
  bool is_catch)
{
  irep_idt identifier = type.identifier();
  const typet t = ns.lookup(identifier)->get_type();

  if (t.id() != "struct" || is_catch)
  {
    ids.emplace_back(id2string(identifier) + suffix);
    return;
  }

  // The `tag-` prefix is stripped: an exception id names the class, not its
  // type symbol.
  ids.emplace_back(id2string(identifier).substr(4) + suffix);

  const exprt &bases =
    static_cast<const exprt &>(to_struct_type(t).find("bases"));
  if (bases.is_nil())
    return;

  for (const auto &i : bases.get_sub())
    ids.emplace_back(id2string(i.id()).substr(4) + suffix);
}

/// The C++ spelling, when the type carries one, and a last-resort id so the
/// list is never empty: callers such as adjust_catch dereference `ids.front()`,
/// and an unusual catch parameter (a function type, as in the ill-formed
/// `catch (exception())`) matches none of the cases above. A synthetic id never
/// matches a real throw, which is the intended behaviour there.
static void append_cpp_spelling_and_fallback(
  const typet &type,
  const std::string &suffix,
  std::vector<irep_idt> &ids)
{
  const std::string cpp_type = type.cpp_type().as_string();
  if (!cpp_type.empty())
    ids.emplace_back(cpp_type + suffix);

  if (ids.empty())
    ids.emplace_back(id2string(type.id()) + suffix);
}

void convert_exception_id(
  const namespacet &ns,
  const typet &type,
  const std::string &suffix,
  std::vector<irep_idt> &ids,
  bool is_catch)
{
  if (type.id() == "pointer" || type.id() == "array")
  {
    if (type.reference())
    {
      convert_exception_id(ns, type.subtype(), suffix, ids, is_catch);
      return;
    }
    if (type.subtype().id() == "empty")
    {
      irep_idt identifier = "void_ptr";
      ids.emplace_back(id2string(identifier) + suffix);
    }
    else
    {
      convert_exception_id(ns, type.subtype(), "_ptr" + suffix, ids, is_catch);
      return;
    }
  }
  else if (type.id() == "struct")
  {
    // An aggregate-initialised thrown object (`throw E{...}`, no constructor)
    // arrives with an inline struct type rather than a symbol reference,
    // because get_complete_type resolves it during InitListExpr conversion.
    // Resolve it back to the class's type symbol so its exception id matches a
    // `catch (E)` clause's symbol-typed id; otherwise the throw and the handler
    // disagree and the exception escapes uncaught (#6300).
    irep_idt name = type.get("name");
    const symbolt *sym = name.empty() ? nullptr : ns.lookup(name);
    if (sym == nullptr && !type.get("tag").as_string().empty())
      sym = ns.lookup("tag-" + type.get("tag").as_string());
    if (sym != nullptr && sym->get_type().id() == "struct")
    {
      symbol_typet sym_type(sym->id);
      convert_exception_id(ns, sym_type, suffix, ids, is_catch);
      return;
    }
    ids.emplace_back(id2string(type.id()) + suffix);
  }
  else if (type.id() == "symbol")
    class_exception_ids(ns, type, suffix, ids, is_catch);
  else if (type.ellipsis())
  {
    irep_idt identifier = "ellipsis";
    ids.emplace_back(id2string(identifier) + suffix);
  }
  else if (type.id() == "noexcept")
  {
    irep_idt identifier = "noexcept";
    ids.emplace_back(id2string(identifier) + suffix);
  }

  append_cpp_spelling_and_fallback(type, suffix, ids);
}
