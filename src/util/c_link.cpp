#include <clang-c-frontend/expr2c.h>
#include <unordered_set>
#include <util/base_type.h>
#include <util/c_link.h>
#include <util/expr_util.h>
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
    const symbolt *t = second.lookup(name);
    if(!s)
      return t;
    if(!t)
      return s;
    if(s->odr_override != t->odr_override)
      return s->odr_override ? s : t;
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
  bool duplicate_type(symbolt &in_context, symbolt &new_symbol);
  bool duplicate_symbol(symbolt &in_context, symbolt &new_symbol);
  symbolt *move(symbolt &new_symbol);

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

  bool context_needs_fixing = false;
};

std::string c_linkt::to_string(const exprt &expr)
{
  return expr2c(expr, ns);
}

std::string c_linkt::to_string(const typet &type)
{
  return type2c(type, ns);
}

void c_linkt::duplicate(symbolt &in_context, symbolt &new_symbol)
{
  if(new_symbol.is_type != in_context.is_type)
  {
    log_error("class conflict on symbol `{}'", in_context.name);
    abort();
  }

  if(new_symbol.is_type && duplicate_type(in_context, new_symbol))
  {
    symbol_fixer.insert(in_context.id, new_symbol.type);
    context_needs_fixing = true;
  }
  else if(duplicate_symbol(in_context, new_symbol))
  {
    symbol_fixer.insert(in_context.id, new_symbol.value);
    context_needs_fixing = true;
  }
}

bool c_linkt::duplicate_type(symbolt &in_context, symbolt &new_symbol)
{
  auto worse = [this](const symbolt &fst, const symbolt &snd) {
    irep_idt a = ns.follow(fst.type).id(), b = ns.follow(snd.type).id();
    return (a == "incomplete_struct" && b == "struct") ||
           (a == "incomplete_union" && b == "union") ||
           (a == "incomplete_array" && b == "array") ||
           (!fst.odr_override && snd.odr_override);
  };

  // check if it is the same -- use base_type_eq
  if(base_type_eq(in_context.type, new_symbol.type, ns))
    return false;

  if(worse(in_context, new_symbol))
  {
    // replace old symbol
    in_context = new_symbol;
    return true;
  }

  if(worse(new_symbol, in_context))
  {
    // ignore
    return false;
  }

  // rename, there are no type clashes in C
  irep_idt old_identifier = new_symbol.id;

  do
  {
    irep_idt new_identifier =
      id2string(old_identifier) + "#link" + i2string(type_counter++);

    new_symbol.id = new_identifier;
  } while(context.move(new_symbol));

  return true;
}

bool c_linkt::duplicate_symbol(symbolt &in_context, symbolt &new_symbol)
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

    if(new_symbol.value.is_nil())
      return false;

    if(in_context.value.is_nil())
    {
      // the one with body wins!
      in_context.value.swap(new_symbol.value);
      in_context.type.swap(new_symbol.type); // for argument identifiers
      return false;
    }
    if(in_context.type.inlined())
    {
      // ok
      return false;
    }
    if(base_type_eq(in_context.type, new_symbol.type, ns))
    {
      // keep the one in in_context -- libraries come last!
      log_warning(
        "warning: function `{}' in module `{}' "
        "is shadowed by a definition in module `{}'",
        in_context.name,
        new_symbol.module,
        in_context.module);
      return false;
    }

    log_error(
      "error: duplicate definition of function `{}'\n"
      "In module `{}' and module `{}'\n"
      "Location: {}",
      in_context.name,
      in_context.module,
      new_symbol.value.location());
    abort();
  }
  else
  {
    // both are variables
    bool changed = false;

    if(!base_type_eq(in_context.type, new_symbol.type, ns))
    {
      const typet &old_type = ns.follow(in_context.type);
      const typet &new_type = ns.follow(new_symbol.type);

      if(old_type.is_incomplete_array() && new_type.is_array())
      {
        // store new type
        in_context.type = new_symbol.type;
        changed = true;
      }
      else if(old_type.is_pointer() && new_type.is_array())
      {
        // store new type
        in_context.type = new_symbol.type;
        changed = true;
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
        changed = true;
      }
      else if(old_type.is_struct() && new_type.id() == "incomplete_struct")
      {
        // ignore
      }
      else if(old_type.id() == "incomplete_union" && new_type.is_union())
      {
        // store new type
        in_context.type = new_symbol.type;
        changed = true;
      }
      else if(old_type.is_union() && new_type.id() == "incomplete_union")
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

    return changed;
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

  context.Foreach_operand([this](symbolt &s) {
    if(!s.is_type && s.value.zero_initializer())
    {
      s.value = gen_zero(ns.follow(s.type, true), true);
      s.value.zero_initializer(true);
    }
  });

  if(context_needs_fixing)
    context.Foreach_operand([this](symbolt &s) {
#if 1
      symbol_fixer.fix_symbol(s);
      /* For the types having static initializers defined, replace any such
       * static initialization */
      if(!s.is_type &&
         (s.type.tag() == "pthread_mutex_t" ||
          s.type.tag() == "pthread_cond_t" ||
          s.type.tag() == "pthread_rwlock_t" ||
          s.type.tag() == "pthread_once_t") &&
         s.static_lifetime && !s.value.zero_initializer())
      {
        assert(s.value.is_zero(true));
        s.value = gen_zero(ns.follow(s.type, true), true);
      }
#else
      symbol_fixer.replace(s.type);
      if(!s.is_type)
        symbol_fixer.replace(s.value);
#endif
    });
}

symbolt *c_linkt::move(symbolt &new_symbol)
{
  // try to add it

  symbolt *new_symbol_ptr;
  if(context.move(new_symbol, new_symbol_ptr))
    duplicate(*new_symbol_ptr, new_symbol);

  return new_symbol_ptr;
}
} /* end anonymous namespace */

#include <fstream>

bool c_link(contextt &context, contextt &new_context, const std::string &module)
{
  static size_t link_no = 0;
  link_no++;

  struct dump
  {
    mutable std::ofstream out;

    dump(const char *prefix, const contextt &ctx)
      : out(prefix + std::to_string(link_no))
    {
      ctx.foreach_operand(*this);
    }

    void operator()(const symbolt &s) const
    {
      s.show(out);
      out << "\n---------------------------------------------------------------"
             "-------------\n";
    }
  };

  if(messaget::state.verbosity >= VerbosityLevel::Debug)
    dump("tgt-", context);
  if(messaget::state.verbosity >= VerbosityLevel::Debug)
    dump("src-", new_context);

  c_linkt c_link(context, new_context, module);
  bool r = c_link.typecheck_main();

  if(messaget::state.verbosity >= VerbosityLevel::Debug)
    dump("res-", context);

  return r;
}
