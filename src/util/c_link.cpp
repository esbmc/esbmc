#include <util/c_expr2string.h>
#include <unordered_set>
#include <util/base_type.h>
#include <util/c_link.h>
#include <util/fix_symbol.h>
#include <util/i2string.h>
#include <util/location.h>
#include <util/message/format.h>
#include <util/namespace.h>
#include <util/typecheck.h>

namespace
{
class merged_namespacet : public namespacet
{
  namespacet second;

public:
  merged_namespacet(const contextt &primary, const contextt &secondary)
    : namespacet(primary), second(secondary)
  {
  }

  unsigned get_max(const std::string &prefix) const override
  {
    return std::max(namespacet::get_max(prefix), second.get_max(prefix));
  }

  const symbolt *lookup(const irep_idt &name) const override
  {
    const symbolt *s = namespacet::lookup(name);
    if(!s)
      s = second.lookup(name);
    return s;
  }
};

class c_linkt : public typecheckt
{
public:
  c_linkt(contextt &_context, contextt &_new_context, std::string _module)
    : context(_context),
      new_context(_new_context),
      module(std::move(_module)),
      ns(_context, _new_context),
      type_counter(0)
  {
    context.Foreach_operand([this](symbolt &s) {
      if(!s.module.empty())
        known_modules.insert(s.module);
    });
  }

  void typecheck() override;

  /**
   * @brief Merges extern symbols from old contexts
   *
   * Checks whether a non-extern new_context symbol
   * was an extern symbol in a previous context.
   *
   * If it was, then merge it.
   *
   * @param s
   */
  void extern_fixup(symbolt &s);

protected:
  void duplicate(symbolt &in_context, symbolt &new_symbol);
  void duplicate_type(symbolt &in_context, symbolt &new_symbol);
  void duplicate_symbol(symbolt &in_context, symbolt &new_symbol);
  void move(symbolt &new_symbol);

  // overload to use language specific syntax
  std::string to_string(const exprt &expr);
  std::string to_string(const typet &type);

  contextt &context;
  contextt &new_context;
  std::string module;
  merged_namespacet ns;

  typedef std::unordered_set<irep_idt, irep_id_hash> known_modulest;
  known_modulest known_modules;

  fix_symbolt symbol_fixer;

  unsigned type_counter;
};

std::string c_linkt::to_string(const exprt &expr)
{
  return c_expr2string(expr, ns);
}

std::string c_linkt::to_string(const typet &type)
{
  return c_type2string(type, ns);
}

void c_linkt::duplicate(symbolt &in_context, symbolt &new_symbol)
{
  if(new_symbol.is_type != in_context.is_type)
  {
    log_error("class conflict on symbol `{}'", in_context.name);
    abort();
  }

  if(new_symbol.is_type)
    duplicate_type(in_context, new_symbol);
  else
    duplicate_symbol(in_context, new_symbol);
}

void c_linkt::duplicate_type(symbolt &in_context, symbolt &new_symbol)
{
  // check if it is the same -- use base_type_eq
  if(!base_type_eq(in_context.type, new_symbol.type, ns))
  {
    if(
      in_context.type.id() == "incomplete_struct" &&
      new_symbol.type.id() == "struct")
    {
      // replace old symbol
      in_context.type = new_symbol.type;
    }
    else if(
      in_context.type.id() == "struct" &&
      new_symbol.type.id() == "incomplete_struct")
    {
      // ignore
    }
    else if(
      ns.follow(in_context.type).id() == "incomplete_array" &&
      ns.follow(new_symbol.type).is_array())
    {
      // store new type
      in_context.type = new_symbol.type;
    }
    else if(
      ns.follow(in_context.type).is_array() &&
      ns.follow(new_symbol.type).id() == "incomplete_array")
    {
      // ignore
    }
    else
    {
      // rename, there are no type clashes in C
      irep_idt old_identifier = new_symbol.id;

      do
      {
        irep_idt new_identifier =
          id2string(old_identifier) + "#link" + i2string(type_counter++);

        new_symbol.id = new_identifier;
      } while(context.move(new_symbol));
    }
  }
}

void c_linkt::duplicate_symbol(symbolt &in_context, symbolt &new_symbol)
{
  // see if it is a function or a variable

  bool is_code_in_context = in_context.type.is_code();
  bool is_code_new_symbol = new_symbol.type.is_code();

  if(is_code_in_context != is_code_new_symbol)
  {
    log_error(
      "error: conflicting definition for symbol \"{}\"\n"
      "old definition: {}\n"
      "Module: {}\n"
      "new definition: {}\n"
      "Module: {}",
      in_context.name,
      to_string(in_context.type),
      in_context.module,
      to_string(new_symbol.type),
      new_symbol.module);
    abort();
  }

  if(is_code_in_context)
  {
    // both are functions

    // we don't compare the types, they will be too different

    // care about code

    if(!new_symbol.value.is_nil())
    {
      if(in_context.value.is_nil())
      {
        // the one with body wins!
        in_context.value.swap(new_symbol.value);
        in_context.type.swap(new_symbol.type); // for argument identifiers
      }
      else if(in_context.type.inlined())
      {
        // ok
      }
      else if(base_type_eq(in_context.type, new_symbol.type, ns))
      {
        // keep the one in in_context -- libraries come last!
        log_warning(
          "function `{}' in module `{}' is shadowed by a definition in module "
          "`{}'",
          in_context.name,
          new_symbol.module,
          in_context.module);
      }
      else
      {
        log_error(
          "error: duplicate definition of function `{}'\n"
          "In module `{}' and module `{}'\n"
          "Location: {}",
          in_context.name,
          in_context.module,
          new_symbol.module,
          new_symbol.value.location());
      }
    }
  }
  else
  {
    // both are variables

    if(!base_type_eq(in_context.type, new_symbol.type, ns))
    {
      const typet &old_type = ns.follow(in_context.type);
      const typet &new_type = ns.follow(new_symbol.type);

      if(old_type.is_incomplete_array() && new_type.is_array())
      {
        // store new type
        in_context.type = new_symbol.type;
      }
      else if(old_type.is_pointer() && new_type.is_array())
      {
        // store new type
        in_context.type = new_symbol.type;
      }
      else if(old_type.is_array() && new_type.is_pointer())
      {
        // ignore
      }
      else if(old_type.is_array() && new_type.is_incomplete_array())
      {
        // ignore
      }
      else if(old_type.id() == "incomplete_struct" && new_type.is_struct())
      {
        // store new type
        in_context.type = new_symbol.type;
      }
      else if(old_type.is_struct() && new_type.id() == "incomplete_struct")
      {
        // ignore
      }
      else if(old_type.is_pointer() && new_type.is_incomplete_array())
      {
        // ignore
      }
#ifdef _WIN32
      // Windows is not case-sensitive
      else if(in_context.module.compare_uppercase(new_symbol.module))
      {
        // ignore
      }
#endif
      else
      {
        log_error(
          "error: conflicting definition for variable `{}'\n"
          "old definition: {}\n"
          "Module: {}\n"
          "new definition: {}\n"
          "Module: {}\n"
          "Location: {}",
          in_context.name,
          to_string(in_context.type),
          in_context.module,
          to_string(new_symbol.type),
          new_symbol.module,
          new_symbol.location);
      }
    }

    // care about initializers

    if(!new_symbol.value.is_nil() && !new_symbol.value.zero_initializer())
    {
      if(in_context.value.is_nil() || in_context.value.zero_initializer())
      {
        in_context.value.swap(new_symbol.value);
      }
      else if(!base_type_eq(in_context.value, new_symbol.value, ns))
      {
        log_error(
          "error: conflicting initializers for variable `{}'\n"
          "old value: {}\n"
          "Module: {}\n"
          "new value: {}\n"
          "Module: {}",
          in_context.name,
          to_string(in_context.value),
          in_context.module,
          to_string(new_symbol.value),
          new_symbol.module);
        abort();
      }
    }
  }
}

void c_linkt::extern_fixup(symbolt &s)
{
  if(!s.is_extern)
  {
    // If the previous context had it
    auto prev = context.find_symbol(s.id);
    if(prev)
    {
      // If current context is not extern and previous was
      if(!s.is_extern && prev->is_extern)
        prev->swap(s);
    }
  }
}

void c_linkt::typecheck()
{
  new_context.Foreach_operand([this](symbolt &s) {
    // First, if the symbol is extern, then check whether it can be merged
    extern_fixup(s);
    // build module clash table
    if(s.file_local && known_modules.find(s.module) != known_modules.end())
    {
      // we could have a clash
      unsigned counter = 0;
      std::string newname = id2string(s.id);

      while(context.find_symbol(newname) != nullptr)
      {
        // there is a clash, rename!
        counter++;
        newname = id2string(s.id) + "#-mc-" + i2string(counter);
      }

      if(counter > 0)
      {
        exprt subst("symbol");
        subst.identifier(newname);
        subst.location() = s.location;
        symbol_fixer.insert(
          s.id, static_cast<const typet &>(static_cast<const irept &>(subst)));
        subst.type() = s.type;
        symbol_fixer.insert(s.id, subst);
      }
    }
  });

  symbol_fixer.fix_context(new_context);

  new_context.Foreach_operand_in_order([this](symbolt &s) {
    symbol_fixer.fix_symbol(s);
    move(s);
  });
}

void c_linkt::move(symbolt &new_symbol)
{
  // try to add it

  symbolt *new_symbol_ptr;
  if(context.move(new_symbol, new_symbol_ptr))
    duplicate(*new_symbol_ptr, new_symbol);
}
} /* end anonymous namespace */

bool c_link(contextt &context, contextt &new_context, const std::string &module)
{
  c_linkt c_link(context, new_context, module);
  return c_link.typecheck_main();
}
