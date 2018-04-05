/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/cpp_convert_type.h>
#include <cpp/cpp_template_type.h>
#include <cpp/cpp_type2name.h>
#include <cpp/cpp_typecheck.h>
#include <cpp/cpp_typecheck_resolve.h>
#include <cpp/cpp_util.h>
#include <util/arith_tools.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/prefix.h>
#include <util/std_expr.h>
#include <util/std_types.h>
#include <util/string_constant.h>
#include <utility>

cpp_typecheck_resolvet::cpp_typecheck_resolvet(cpp_typecheckt &_cpp_typecheck)
  : cpp_typecheck(_cpp_typecheck)
{
}

void cpp_typecheck_resolvet::convert_identifiers(
  const cpp_scopest::id_sett &id_set,
  const wantt want,
  const cpp_typecheck_fargst &fargs,
  resolve_identifierst &identifiers)
{
  for(auto it : id_set)
  {
    const cpp_idt &identifier = *it;
    exprt e = convert_identifier(identifier, want, fargs);

    if(e.is_not_nil())
    {
      if(e.id() == "type")
        assert(e.type().is_not_nil());

      identifiers.push_back(e);
    }
  }
}

void cpp_typecheck_resolvet::apply_template_args(
  resolve_identifierst &identifiers,
  const cpp_template_args_non_tct &template_args,
  const cpp_typecheck_fargst &fargs)
{
  resolve_identifierst old_identifiers;
  old_identifiers.swap(identifiers);

  for(resolve_identifierst::const_iterator it = old_identifiers.begin();
      it != old_identifiers.end();
      it++)
  {
    exprt e = *it;
    apply_template_args(e, template_args, fargs);

    if(e.is_not_nil())
    {
      if(e.id() == "type")
        assert(e.type().is_not_nil());

      identifiers.push_back(e);
    }
  }
}

void cpp_typecheck_resolvet::guess_function_template_args(
  resolve_identifierst &identifiers,
  const cpp_typecheck_fargst &fargs)
{
  resolve_identifierst old_identifiers;
  resolve_identifierst non_template_identifiers;
  old_identifiers.swap(identifiers);

  for(resolve_identifierst::const_iterator it = old_identifiers.begin();
      it != old_identifiers.end();
      it++)
  {
    exprt e = guess_function_template_args(*it, fargs);

    if(e.is_not_nil())
    {
      assert(e.id() != "type");
      identifiers.push_back(e);
    }
    else
    {
      if(it->type().get("is_template") == "1")
        // Failed template arg guessing. Discard.
        ;
      else
        // Not a template; save it for later disambiguation.
        non_template_identifiers.push_back(*it);
    }
  }

  remove_duplicates(identifiers);

  // Don't disambiguate functions -- member functions don't have a 'this'
  // parameter until they're instantiated, and without that detail we might pick
  // the wrong template, due to not being able to overload on the 'this'
  // parameter. Instead, instantiate them all, and let a later disambiguation
  // solve tihs problem. SFINAE should prevent any substitution errors
  // manifesting.
  for(auto e : identifiers)
  {
    assert(e.id() == "template_function_instance");

    const symbolt &template_symbol =
      cpp_typecheck.lookup(e.type().get("#template"));

    const cpp_template_args_tct &template_args =
      to_cpp_template_args_tc(e.type().find("#template_arguments"));

    // Let's build the instance.

    const symbolt &new_symbol = cpp_typecheck.instantiate_template(
      location, template_symbol, template_args, template_args);

    // Mark this template as having been instantiated speculatively. This is
    // vital to support SFINAE: when the instantiated template is typechecked
    // later, it may very well have an error in it. We need to know at that
    // point (or beforehand) whether it was speculatively instantiated, and thus
    // might not actually be used.
    // A template that's speculatively instantiated, and used, and contains an
    // error, should still be registered as an error though.
    symbolt &mutable_symbol = const_cast<symbolt &>(new_symbol);
    mutable_symbol.value.set("#speculative_template", "1");

    if(!fargs.has_object || e.type().get("return_type") == "constructor")
    {
      non_template_identifiers.push_back(
        symbol_exprt(new_symbol.name, new_symbol.type));
    }
    else
    {
      // This should be a member expression.
      exprt memb("member");
      memb.type() = new_symbol.type;
      memb.set("component_name", new_symbol.name);
      non_template_identifiers.push_back(memb);
    }
  }

  // Restore the non-template identifiers, which we haven't altered.
  identifiers.swap(non_template_identifiers);
}

void cpp_typecheck_resolvet::remove_templates(resolve_identifierst &identifiers)
{
  resolve_identifierst old_identifiers;
  old_identifiers.swap(identifiers);

  for(resolve_identifierst::const_iterator it = old_identifiers.begin();
      it != old_identifiers.end();
      it++)
  {
    if(!cpp_typecheck.follow(it->type()).get_bool("is_template"))
      identifiers.push_back(*it);
  }
}

void cpp_typecheck_resolvet::remove_duplicates(
  resolve_identifierst &identifiers)
{
  resolve_identifierst old_identifiers;
  old_identifiers.swap(identifiers);

  std::set<irep_idt> ids;
  std::set<exprt> other;

  for(resolve_identifierst::const_iterator it = old_identifiers.begin();
      it != old_identifiers.end();
      it++)
  {
    irep_idt id;

    if(it->id() == "symbol")
      id = it->identifier();
    else if(it->id() == "type" && it->type().id() == "symbol")
      id = it->type().identifier();

    if(id == "")
    {
      if(other.insert(*it).second)
        identifiers.push_back(*it);
    }
    else
    {
      if(ids.insert(id).second)
        identifiers.push_back(*it);
    }
  }
}

void cpp_typecheck_resolvet::disambiguate_copy_constructor(
  resolve_identifierst &identifiers)
{
  // C++ has an ambiguity: templates can be defined that will exactly match
  // the signature of the copy constructor in the right circumstances. Section
  // 12.8.3 specifies that templates must not be candidates for copy
  // constructors, and if anything else is an equal match as the copy
  // constructor, the program is ill formed.
  // To implement this, if the list of identifiers we have contains a copy
  // constructor, go through and remove all templates. That implements the
  // "no-templates" part of the spec.
  // Leaving all other candidates in, even if there are multiple candidates,
  // will implement the "ill formed if any methods equivalent to copy
  // constructor" part. (Because this will be rejected later).

  // First: is there a copy constructor in here?
  bool has_copy_cons = false;
  for(resolve_identifierst::const_iterator it = identifiers.begin();
      it != identifiers.end();
      it++)
  {
    if(
      it->type().id() == "code" &&
      it->type().return_type().get("#default_copy_cons") == "1")
      has_copy_cons = true;
  }

  if(!has_copy_cons)
    return;

  // Erase anything that was a template.

  resolve_identifierst::iterator it = identifiers.begin();
  while(it != identifiers.end())
  {
    if(it->id() != "symbol" && it->id() != "member")
    {
      it++;
      continue;
    }

    // Identify and eliminate templates by seeing if they've been instantiated
    // in apply/guess template args. This is indicated by the speculative
    // template flag.
    const irep_idt &name =
      (it->id() == "symbol") ? it->identifier() : it->component_name();
    const symbolt &sym = cpp_typecheck.lookup(name);
    if(sym.value.get("#speculative_template") == "1")
      it = identifiers.erase(it);
    else
      it++;
  }
}

exprt cpp_typecheck_resolvet::convert_template_argument(
  const cpp_idt &identifier)
{
  // Is there an assignment to this template argument in the template map?
  exprt e = cpp_typecheck.template_map.lookup(identifier.identifier);

  if(e.is_nil() || (e.id() == "type" && e.type().is_nil()))
  {
    // No. In that case, see whether we've picked up the template argument from
    // the instantiation scope, which will mean it has a type attached.

    const symbolt &sym = cpp_typecheck.lookup(identifier.identifier);
    exprt e2;
    if(sym.is_type)
    {
      exprt tmp("type");
      tmp.type() = sym.type;
      e2 = tmp;
    }
    else
    {
      e2 = sym.value;
    }

    e2.location() = location;

    if(e2.is_nil() || e2.type().is_nil())
    {
      cpp_typecheck.err_location(location);
      cpp_typecheck.str
        << "internal error: template parameter without instance:" << std::endl
        << identifier << std::endl;
      throw 0;
    }

    return e2;
  }

  // Just return what was in the template map.
  e.location() = location;
  return e;
}

exprt cpp_typecheck_resolvet::convert_identifier(
  const cpp_idt &identifier,
  const wantt want,
  const cpp_typecheck_fargst &fargs)
{
  if(identifier.id_class == cpp_scopet::TEMPLATE_ARGUMENT)
    return convert_template_argument(identifier);

  exprt e;

  if(
    identifier.is_member && !identifier.is_constructor &&
    !identifier.is_static_member)
  {
    // a regular struct or union member

    const symbolt &class_symbol =
      cpp_typecheck.lookup(identifier.class_identifier);

    assert(
      class_symbol.type.id() == "struct" || class_symbol.type.id() == "union");

    const struct_typet &struct_type = to_struct_type(class_symbol.type);

    const exprt component = struct_type.get_component(identifier.identifier);

    const typet &type = component.type();
    assert(type.is_not_nil());

    if(identifier.id_class == cpp_scopet::TYPEDEF)
    {
      e = type_exprt(type);
    }
    else if(identifier.id_class == cpp_scopet::SYMBOL)
    {
      // A non-static, non-type member.
      // There has to be an object.
      e = exprt("member");
      e.component_name(identifier.identifier);
      e.location() = location;

      exprt object;
      object.make_nil();

#if 0
      std::cout << "I: " << identifier.class_identifier
                << " "
                << cpp_typecheck.cpp_scopes.current_scope().this_class_identifier << std::endl;
#endif

      const exprt &this_expr = original_scope->this_expr;

      // find the object of the member expression
      if(class_symbol.type.find("#unnamed_object").is_not_nil())
      {
        cpp_scopet::id_sett id_set;
        cpp_typecheck.cpp_scopes.current_scope().recursive_lookup(
          class_symbol.type.get("#unnamed_object"), id_set);

        assert(id_set.size() == 1);

        object = convert_identifier(**(id_set.begin()), want, fargs);
        assert(object.is_not_nil());
      }
      else if(fargs.has_object)
      {
        // the object is given to us in fargs
        assert(!fargs.operands.empty());
        object = fargs.operands[0];
      }
      else if(
        component.get_bool("is_operator") &&
        fargs.operands.size() ==
          to_code_type(component.type()).arguments().size())
      {
        // turn  'OP(a, b)' into 'a.opratorOP(b)'
        object = fargs.operands.front();
      }
      else if(this_expr.is_not_nil())
      {
        // use this->...
        assert(this_expr.type().id() == "pointer");
        object = exprt("dereference", this_expr.type().subtype());
        object.copy_to_operands(this_expr);
        object.type().set(
          "#constant", this_expr.type().subtype().cmt_constant());
        object.set("#lvalue", true);
        object.location() = location;
      }

      // check if the member can be applied to the object
      typet object_type = cpp_typecheck.follow(object.type());

      if(object_type.id() == "struct" || object_type.id() == "union")
      {
        const struct_typet &object_struct = to_struct_type(object_type);
        if(object_struct.get_component(identifier.identifier.c_str()).is_nil())
          object.make_nil();
      }
      else
        object.make_nil();

      if(object.is_not_nil())
      {
        // we got an object
        e.move_to_operands(object);

        bool old_value = cpp_typecheck.disable_access_control;
        cpp_typecheck.disable_access_control = true;
        cpp_typecheck.typecheck_expr_member(e);
        cpp_typecheck.disable_access_control = old_value;
      }
      else
      {
        // this has to be a method
        if(identifier.is_method)
          e = cpp_symbol_expr(cpp_typecheck.lookup(identifier.identifier));
        else
          e.make_nil();
      }
    }
  }
  else
  {
    const symbolt &symbol = cpp_typecheck.lookup(identifier.identifier);

    if(symbol.is_type)
    {
      e = type_exprt();

      if(symbol.is_macro)
      {
        e.type() = symbol.type;
        assert(symbol.type.is_not_nil());
      }
      else
        e.type() = symbol_typet(symbol.name);
    }
    else if(symbol.is_macro)
    {
      e = symbol.value;
      assert(e.is_not_nil());
    }
    else
    {
      typet followed_type = symbol.type;
      bool constant = followed_type.cmt_constant();

      while(followed_type.id() == "symbol")
      {
        typet tmp = cpp_typecheck.lookup(followed_type).type;
        followed_type = tmp;
        constant |= followed_type.cmt_constant();
      }

      if(
        constant && symbol.value.is_not_nil() && is_number(followed_type) &&
        symbol.value.id() == "constant")
      {
        e = symbol.value;
      }
      else
      {
        e = cpp_symbol_expr(symbol);
      }
    }
  }

  e.location() = location;

  return e;
}

void cpp_typecheck_resolvet::filter(
  resolve_identifierst &identifiers,
  const wantt want)
{
  resolve_identifierst old_identifiers;
  old_identifiers.swap(identifiers);

  for(resolve_identifierst::const_iterator it = old_identifiers.begin();
      it != old_identifiers.end();
      it++)
  {
    bool match = false;

    switch(want)
    {
    case TYPE:
      match = (it->id() == "type");
      break;

    case VAR:
      match = (it->id() != "type");
      break;

    case BOTH:
      match = true;
      break;

    default:
      assert(false);
      break;
    }

    if(match)
      identifiers.push_back(*it);
  }
}

void cpp_typecheck_resolvet::filter(
  cpp_scopest::id_sett &id_set,
  const wantt want)
{
  // When searching for templates named 'foo', any constructor templates for
  // 'foo' will be returned too. This results in the function template code
  // becoming unhappy for reasons unknown, and encoding template arguments into
  // the symbol table twice (boom).
  // Fix this by, when searching for types, eliminating function templates.

  if(want != TYPE)
    return;

  cpp_scopest::id_sett old_set;
  old_set.swap(id_set);
  for(auto it : old_set)
  {
    // OK; what kind of template are we dealing with here...
    const symbolt &sym = cpp_typecheck.lookup(it->identifier);
    if(sym.type.type().id() == "struct")
      id_set.insert(it);
  }
}

void cpp_typecheck_resolvet::exact_match_functions(
  resolve_identifierst &identifiers,
  const cpp_typecheck_fargst &fargs)
{
  if(!identifiers.size())
    return;
  if(!fargs.in_use)
    return;

  resolve_identifierst old_identifiers;
  old_identifiers.swap(identifiers);

  identifiers.clear();

  // put in the ones that match precisely
  for(resolve_identifierst::const_iterator it = old_identifiers.begin();
      it != old_identifiers.end();
      it++)
  {
    cpp_typecast_rank distance;
    if(disambiguate_functions(*it, distance, fargs))
      if(distance.rank <= 1)
        identifiers.push_back(*it);
  }
}

void cpp_typecheck_resolvet::disambiguate_functions(
  resolve_identifierst &identifiers,
  const cpp_typecheck_fargst &fargs)
{
  if(identifiers.size() < 2)
    return;

  resolve_identifierst old_identifiers;
  old_identifiers.swap(identifiers);

  // sort according to distance
  std::multimap<cpp_typecast_rank, exprt> distance_map;

  for(resolve_identifierst::const_iterator it = old_identifiers.begin();
      it != old_identifiers.end();
      it++)
  {
    cpp_typecast_rank args_distance;

    if(disambiguate_functions(*it, args_distance, fargs))
    {
      unsigned template_distance = 0;

      if(it->type().get("#template") != "")
        template_distance =
          it->type().find("#template_arguments").arguments().get_sub().size();

      // we give strong preference to functions that have
      // fewer template arguments
      args_distance.templ_distance += template_distance;

      distance_map.insert(std::make_pair(args_distance, *it));
    }
  }

  identifiers.clear();

  // put in the top ones
  if(!distance_map.empty())
  {
    const cpp_typecast_rank &distance = distance_map.begin()->first;

    for(std::multimap<cpp_typecast_rank, exprt>::const_iterator it =
          distance_map.begin();
        // "While rank not worse that then start of the lists rank"
        it != distance_map.end() && !(distance < it->first);
        it++)
      identifiers.push_back(it->second);
  }

  if(identifiers.size() > 1 && fargs.in_use)
  {
    // try to further disambiguate functions

    for(resolve_identifierst::iterator it1 = identifiers.begin();
        it1 != identifiers.end();
        it1++)
    {
      if(it1->type().id() != "code")
        continue;

      const code_typet &f1 = to_code_type(it1->type());

      for(resolve_identifierst::iterator it2 = identifiers.begin();
          it2 != identifiers.end();) // no it2++
      {
        if(it1 == it2)
        {
          it2++;
          continue;
        }

        if(it2->type().id() != "code")
        {
          it2++;
          continue;
        }

        const code_typet &f2 = to_code_type(it2->type());

        // TODO: may fail when using ellipsis
        assert(f1.arguments().size() == f2.arguments().size());

        bool f1_better = true;
        bool f2_better = true;

        for(unsigned i = 0;
            i < f1.arguments().size() && (f1_better || f2_better);
            i++)
        {
          typet type1 = f1.arguments()[i].type();
          typet type2 = f2.arguments()[i].type();

          if(type1 == type2)
            continue;

          if(is_reference(type1) != is_reference(type2))
            continue;

          if(type1.id() == "pointer")
          {
            typet tmp = type1.subtype();
            type1 = tmp;
          }

          if(type2.id() == "pointer")
          {
            typet tmp = type2.subtype();
            type2 = tmp;
          }

          const typet &followed1 = cpp_typecheck.follow(type1);
          const typet &followed2 = cpp_typecheck.follow(type2);

          if(followed1.id() != "struct" || followed2.id() != "struct")
            continue;

          const struct_typet &struct1 = to_struct_type(followed1);
          const struct_typet &struct2 = to_struct_type(followed2);

          if(f1_better && cpp_typecheck.subtype_typecast(struct1, struct2))
          {
            f2_better = false;
          }
          else if(f2_better && cpp_typecheck.subtype_typecast(struct2, struct1))
          {
            f1_better = false;
          }
        }

        resolve_identifierst::iterator prev_it = it2;
        it2++;

        if(f1_better && !f2_better)
          identifiers.erase(prev_it);
      }
    }
  }
}

void cpp_typecheck_resolvet::make_constructors(
  resolve_identifierst &identifiers)
{
  resolve_identifierst new_identifiers;

  resolve_identifierst::iterator next;

  for(auto &identifier : identifiers)
  {
    if(identifier.id() != "type")
    {
      // already an expression
      new_identifiers.push_back(identifier);
      continue;
    }

    const typet &symbol_type = cpp_typecheck.follow(identifier.type());

    if(symbol_type.id() != "struct")
    {
      // it's ok
      new_identifiers.push_back(identifier);
      continue;
    }

    if(cpp_typecheck.cpp_is_pod(symbol_type))
    {
      // in that case, there is no constructor to call
      new_identifiers.push_back(identifier);
      continue;
    }

    struct_typet struct_type = to_struct_type(symbol_type);

    {
      cpp_save_scopet cpp_saved_scope(cpp_typecheck.cpp_scopes);
      cpp_typecheck.cpp_scopes.set_scope(struct_type.name());
      const symbolt &the_sym = cpp_typecheck.lookup(struct_type.name());

      cpp_scopest::id_sett id_set;
      cpp_typecheck.cpp_scopes.get_ids(the_sym.base_name, id_set, true);

      for(auto it : id_set)
      {
        const symbolt &sub_sym = cpp_typecheck.lookup(it->identifier);

        // Pick out member expressions that are constructors
        if(
          sub_sym.type.id() == "code" &&
          sub_sym.type.return_type().id() == "constructor")
          new_identifiers.push_back(cpp_symbol_expr(sub_sym));

        // Also template that are constructors.
        if(
          sub_sym.type.id() == "cpp-declaration" &&
          sub_sym.type.type().id() == "constructor")
          new_identifiers.push_back(cpp_symbol_expr(sub_sym));
      }
    }
  }

  identifiers = new_identifiers;
}

exprt cpp_typecheck_resolvet::do_builtin(
  const irep_idt &base_name,
  const cpp_template_args_non_tct &template_args)
{
  exprt dest;

  const cpp_template_args_non_tct::argumentst &arguments =
    template_args.arguments();

  if(base_name == "unsignedbv" || base_name == "signedbv")
  {
    if(arguments.size() != 1)
    {
      cpp_typecheck.err_location(location);
      throw id2string(base_name) +
        " expects one template argument, "
        "but got " +
        i2string((unsigned long)arguments.size());
    }

    const exprt &argument = arguments[0];

    if(argument.id() == "type")
    {
      cpp_typecheck.err_location(location);
      throw id2string(base_name) +
        " expects one integer template argument, "
        "but got type";
    }

    mp_integer i;
    if(to_integer(argument, i))
    {
      cpp_typecheck.err_location(location);
      throw "template argument must be constant";
    }

    if(i < 1)
    {
      cpp_typecheck.err_location(location);
      throw "template argument must be greater than zero";
    }

    dest = type_exprt(typet(base_name));
    dest.type().width(integer2string(i));
  }
  else if(base_name == "fixedbv")
  {
    if(arguments.size() != 2)
    {
      cpp_typecheck.err_location(location);
      throw id2string(base_name) +
        " expects two template arguments, "
        "but got " +
        i2string((unsigned long)arguments.size());
    }

    const exprt &argument0 = arguments[0];
    const exprt &argument1 = arguments[1];

    if(argument0.id() == "type")
    {
      cpp_typecheck.err_location(argument0);
      throw id2string(base_name) +
        " expects two integer template arguments, "
        "but got type";
    }

    if(argument1.id() == "type")
    {
      cpp_typecheck.err_location(argument1);
      throw id2string(base_name) +
        " expects two integer template arguments, "
        "but got type";
    }

    mp_integer width, integer_bits;

    if(to_integer(argument0, width))
    {
      cpp_typecheck.err_location(argument0);
      throw "template argument must be constant";
    }

    if(to_integer(argument1, integer_bits))
    {
      cpp_typecheck.err_location(argument1);
      throw "template argument must be constant";
    }

    if(width < 1)
    {
      cpp_typecheck.err_location(argument0);
      throw "template argument must be greater than zero";
    }

    if(integer_bits < 0)
    {
      cpp_typecheck.err_location(argument1);
      throw "template argument must be greater or equal zero";
    }

    if(integer_bits > width)
    {
      cpp_typecheck.err_location(argument1);
      throw "template argument must be smaller or equal width";
    }

    dest = type_exprt(typet(base_name));
    dest.type().width(integer2string(width));
    dest.type().set("integer_bits", integer2string(integer_bits));
  }
  else if(has_prefix(id2string(base_name), "constant_infinity"))
  {
    // ok, but type missing
    dest = exprt("infinity");
  }
  else if(base_name == "dump_scopes")
  {
    dest = exprt("constant", typet("empty"));
    cpp_typecheck.str << "Scopes in location " << location << std::endl;
    cpp_typecheck.cpp_scopes.get_root_scope().print(cpp_typecheck.str);
    cpp_typecheck.warning();
  }
  else if(base_name == "current_scope")
  {
    dest = exprt("constant", typet("empty"));
    cpp_typecheck.str << "Scope in location " << location << ": "
                      << original_scope->prefix;
    cpp_typecheck.warning();
  }
  else if(base_name == "context")
  {
    dest = exprt("constant", typet("empty"));
    cpp_typecheck.context.dump();
    cpp_typecheck.warning();
  }
  else
  {
    cpp_typecheck.err_location(location);
    cpp_typecheck.str << "unknown built-in identifier: " << base_name;
    throw 0;
  }

  return dest;
}

cpp_scopet &cpp_typecheck_resolvet::resolve_scope(
  const cpp_namet &cpp_name,
  std::string &base_name,
  cpp_template_args_non_tct &template_args)
{
  assert(cpp_name.id() == "cpp-name");
  assert(!cpp_name.get_sub().empty());

  original_scope = &cpp_typecheck.cpp_scopes.current_scope();
  location = cpp_name.location();

  irept::subt::const_iterator pos = cpp_name.get_sub().begin();

  bool recursive = true;

  // check if we need to go to the root scope
  if(pos->id() == "::")
  {
    pos++;
    cpp_typecheck.cpp_scopes.go_to_root_scope();
    recursive = false;
  }

  std::string final_base_name;
  template_args.make_nil();

  while(pos != cpp_name.get_sub().end())
  {
    if(pos->id() == "name")
      final_base_name += pos->get_string("identifier");
    else if(pos->id() == "template_args")
      template_args = to_cpp_template_args_non_tc(*pos);
    else if(pos->id() == "::")
    {
      cpp_scopest::id_sett id_set;

      if(template_args.is_not_nil())
      {
        cpp_typecheck.cpp_scopes.get_ids(
          final_base_name, cpp_idt::TEMPLATE, id_set, !recursive);

        const symbolt &symb_tmpl =
          disambiguate_template_classes(base_name, id_set, template_args);

        cpp_typecheck.cpp_scopes.go_to(
          cpp_typecheck.cpp_scopes.get_scope(symb_tmpl.name));

        template_args.make_nil();
      }
      else
      {
        cpp_typecheck.cpp_scopes.get_ids(final_base_name, id_set, !recursive);

        filter_for_named_scopes(id_set);

        if(id_set.size() == 2)
        {
          // Special case: if in this scope there's a) a template and b) a
          // class, then one is an instantiation of the other. And (only) in
          // this case, if there are no template arguments, resolve to the class
          // instead of the template.
          cpp_scopest::id_sett::iterator it = id_set.begin();
          cpp_scopest::id_sett::iterator it1 = it;
          const cpp_idt &id1 = **it++;
          cpp_scopest::id_sett::iterator it2 = it;
          const cpp_idt &id2 = **it++;

          if(
            (id1.id_class == cpp_idt::TEMPLATE &&
             id2.id_class == cpp_idt::CLASS) ||
            (id1.id_class == cpp_idt::CLASS &&
             id2.id_class == cpp_idt::TEMPLATE))
          {
            if(id1.id_class == cpp_idt::TEMPLATE)
              id_set.erase(it1);
            else
              id_set.erase(it2);
          }
        }

        if(id_set.empty())
        {
          cpp_typecheck.show_instantiation_stack(cpp_typecheck.str);
          cpp_typecheck.err_location(location);
          cpp_typecheck.str << "scope `" << final_base_name << "' not found";
          throw 0;
        }
        if(id_set.size() >= 2)
        {
          cpp_typecheck.show_instantiation_stack(cpp_typecheck.str);
          cpp_typecheck.err_location(location);
          cpp_typecheck.str << "scope `" << final_base_name << "' is ambiguous";
          throw 0;
        }

        assert(id_set.size() == 1);
        cpp_typecheck.cpp_scopes.go_to(**id_set.begin());
      }

      // we start from fresh
      final_base_name.clear();
    }
    else if(pos->id() == "operator")
    {
      final_base_name += "operator";

      irept::subt::const_iterator next = pos + 1;
      assert(next != cpp_name.get_sub().end());

      if(
        next->id() == "cpp-name" || next->id() == "pointer" ||
        next->id() == "int" || next->id() == "char" || next->id() == "bool" ||
        next->id() == "merged_type")
      {
        // it's a cast operator
        irept next_ir = *next;
        typet op_name;
        op_name.swap(next_ir);
        cpp_typecheck.typecheck_type(op_name);
        final_base_name += "(" + cpp_type2name(op_name) + ")";
        pos++;
      }
    }
    else
      final_base_name += pos->id_string();

    pos++;
  }

  base_name = final_base_name;

  return cpp_typecheck.cpp_scopes.current_scope();
}

const symbolt &cpp_typecheck_resolvet::disambiguate_template_classes(
  const irep_idt &base_name,
  const cpp_scopest::id_sett &id_set,
  const cpp_template_args_non_tct &full_template_args)
{
  if(id_set.empty())
  {
    cpp_typecheck.show_instantiation_stack(cpp_typecheck.str);
    cpp_typecheck.err_location(location);
    cpp_typecheck.str << "template scope `" << base_name << "' not found";
    throw 0;
  }

  std::set<irep_idt> primary_templates;

  for(auto it : id_set)
  {
    const irep_idt id = it->identifier;

    const symbolt &s = cpp_typecheck.lookup(id);
    if(!s.type.get_bool("is_template"))
      continue;

    const cpp_declarationt &cpp_declaration = to_cpp_declaration(s.type);
    if(!cpp_declaration.is_class_template())
      continue;

    irep_idt specialization_of = cpp_declaration.get_specialization_of();
    if(specialization_of != "")
      primary_templates.insert(specialization_of);
    else
      primary_templates.insert(id);
  }

  assert(primary_templates.size() != 0);

  if(primary_templates.size() >= 2)
  {
    cpp_typecheck.show_instantiation_stack(cpp_typecheck.str);
    cpp_typecheck.err_location(location);
    cpp_typecheck.str << "template scope `" << base_name << "' is ambiguous";
    throw 0;
  }

  assert(primary_templates.size() == 1);

  const symbolt &primary_template_symbol =
    cpp_typecheck.lookup(*primary_templates.begin());

  // We typecheck the template arguments in the context
  // of the original scope!
  cpp_template_args_tct full_template_args_tc;

  {
    cpp_save_scopet save_scope(cpp_typecheck.cpp_scopes);

    cpp_typecheck.cpp_scopes.go_to(*original_scope);

    // use template type of 'primary template'
    full_template_args_tc = cpp_typecheck.typecheck_template_args(
      location, primary_template_symbol, full_template_args);
    // go back to where we used to be
  }

  // find any matches

  std::vector<matcht> matches;

  // the baseline
  matches.emplace_back(
    full_template_args_tc, full_template_args_tc, primary_template_symbol.name);

  for(auto it : id_set)
  {
    const irep_idt id = it->identifier;
    const symbolt &s = cpp_typecheck.lookup(id);

    irep_idt specialization_of = s.type.get("specialization_of");
    if(specialization_of == "")
      continue;

    const cpp_declarationt &cpp_declaration = to_cpp_declaration(s.type);

    const cpp_template_args_non_tct &partial_specialization_args =
      cpp_declaration.partial_specialization_args();

    // alright, set up template arguments as 'unassigned'

    cpp_saved_template_mapt saved_map(cpp_typecheck.template_map);

    cpp_typecheck.template_map.build_unassigned(
      cpp_declaration.template_type());

    // iterate over template instance
    assert(
      full_template_args_tc.arguments().size() ==
      partial_specialization_args.arguments().size());

    // we need to do this in the right scope

    cpp_scopet *template_scope =
      static_cast<cpp_scopet *>(cpp_typecheck.cpp_scopes.id_map[id]);

    if(template_scope == nullptr)
    {
      cpp_typecheck.err_location(location);
      cpp_typecheck.str << "template identifier: " << id << std::endl;
      throw "class template instantiation error";
    }

    // enter the scope of the template
    cpp_typecheck.cpp_scopes.go_to(*template_scope);

    for(unsigned i = 0; i < full_template_args_tc.arguments().size(); i++)
    {
      if(full_template_args_tc.arguments()[i].id() == "type")
        guess_template_args(
          partial_specialization_args.arguments()[i].type(),
          full_template_args_tc.arguments()[i].type());
      else
        guess_template_args(
          partial_specialization_args.arguments()[i],
          full_template_args_tc.arguments()[i]);
    }

    // see if that has worked out

    cpp_template_args_tct guessed_template_args =
      cpp_typecheck.template_map.build_template_args(
        cpp_declaration.template_type());

    if(!guessed_template_args.has_unassigned())
    {
      // check: we can now typecheck the partial_specialization_args

      cpp_template_args_tct partial_specialization_args_tc =
        cpp_typecheck.typecheck_template_args(
          location, primary_template_symbol, partial_specialization_args);

      // if these match the arguments, we have a match

      assert(
        partial_specialization_args_tc.arguments().size() ==
        full_template_args_tc.arguments().size());

      if(partial_specialization_args_tc == full_template_args_tc)
      {
        matches.emplace_back(
          guessed_template_args, partial_specialization_args_tc, id);
      }
    }
  }

  assert(!matches.empty());

  std::sort(matches.begin(), matches.end());

#if 0
  for(std::vector<matcht>::const_iterator
      m_it=matches.begin();
      m_it!=matches.end();
      m_it++)
  {
    std::cout << "M: " << m_it->cost
              << " " << m_it->id << std::endl;
  }

  std::cout << std::endl;
#endif

  matcht &match = *matches.begin();

  // Let's check if there is more than one 0 distance match
  // This may happen when there is an template specialization on,
  // for example, char and signed_char
  // To solve that, we'll rely on the #cpp_type field, which keeps
  // the original type name

  std::vector<matcht> zero_distance_matches;

  // Lambda expression to insert on the zero_distance_matches if
  // the cost of each element is zero
  std::copy_if(
    matches.begin(),
    matches.end(),
    std::back_inserter(zero_distance_matches),
    [](matcht const &match) { return match.cost == 0; });

  // Check if there was more than one hit
  if(zero_distance_matches.size() > 1)
  {
    auto new_end = std::remove_if(
      zero_distance_matches.begin(),
      zero_distance_matches.end(),
      [&full_template_args_tc](matcht const &match) {
        // This should be replaced by a clean std::remove_if...
        for(unsigned i = 0; i < full_template_args_tc.arguments().size(); ++i)
        {
          irept full_args_cpp =
            match.full_args.arguments()[i].type().find("#cpp_type");
          irept full_template_args_cpp =
            full_template_args_tc.arguments()[i].type().find("#cpp_type");

          // If we cannot get the #cpp_type or if they are different, we remove it
          // from the vector
          if(!(full_args_cpp != irept() && full_template_args_cpp != irept() &&
               full_args_cpp == full_template_args_cpp))
            return true;
        }

        return false;
      });

    zero_distance_matches.erase(new_end, zero_distance_matches.end());

    if(zero_distance_matches.size() == 1)
      match = zero_distance_matches.at(0);
  }

  const symbolt &choice = cpp_typecheck.lookup(match.id);

  // build instance
  const symbolt &instance = cpp_typecheck.instantiate_template(
    location, choice, match.specialization_args, match.full_args);

  if(
    instance.type.id() != "struct" &&
    instance.type.id() != "incomplete_struct" &&
    instance.type.id() != "symbol") // Recursive template def.
  {
    cpp_typecheck.err_location(location);
    cpp_typecheck.str << "template `" << base_name << "' is not a class";
    throw 0;
  }

  return instance;
}

cpp_scopet &cpp_typecheck_resolvet::resolve_namespace(const cpp_namet &cpp_name)
{
  std::string base_name;
  cpp_template_args_non_tct template_args;
  template_args.make_nil();

  cpp_save_scopet save_scope(cpp_typecheck.cpp_scopes);
  resolve_scope(cpp_name, base_name, template_args);

  const locationt &location = cpp_name.location();
  bool qualified = cpp_name.is_qualified();

  cpp_scopest::id_sett id_set;
  cpp_typecheck.cpp_scopes.get_ids(base_name, id_set, qualified);

  filter_for_namespaces(id_set);

  if(id_set.empty())
  {
    cpp_typecheck.err_location(location);
    cpp_typecheck.str << "namespace `" << base_name << "' not found";
    throw 0;
  }
  if(id_set.size() == 1)
  {
    cpp_idt &id = **id_set.begin();
    return (cpp_scopet &)id;
  }
  else
  {
    cpp_typecheck.err_location(location);
    cpp_typecheck.str << "namespace `" << base_name << "' is ambigous";
    throw 0;
  }
}

void cpp_typecheck_resolvet::show_identifiers(
  const std::string &base_name,
  const resolve_identifierst &identifiers,
  std::ostream &out)
{
  for(const auto &id_expr : identifiers)
  {
    out << "  ";

    if(id_expr.id() == "type")
    {
      out << "type " << cpp_typecheck.to_string(id_expr.type());
    }
    else
    {
      irep_idt id;

      if(id_expr.type().get_bool("is_template"))
        out << "template ";

      if(id_expr.id() == "member")
      {
        out << "member ";
        id = "." + base_name;
      }
      else if(id_expr.id() == "pod_constructor")
      {
        out << "constructor ";
        id = "";
      }
      else if(id_expr.id() == "template_function_instance")
      {
        out << "symbol ";
      }
      else
      {
        out << "symbol ";
        id = cpp_typecheck.to_string(id_expr);
      }

      if(id_expr.type().get_bool("is_template"))
      {
        out << cpp_typecheck.lookup(to_symbol_expr(id_expr)).base_name;
      }
      else if(id_expr.type().id() == "code")
      {
        const code_typet &code_type = to_code_type(id_expr.type());
        const typet &return_type = code_type.return_type();
        const code_typet::argumentst &arguments = code_type.arguments();
        out << cpp_typecheck.to_string(return_type);
        out << " " << id << "(";

        for(code_typet::argumentst::const_iterator it = arguments.begin();
            it != arguments.end();
            it++)
        {
          const typet &argument_type = it->type();

          if(it != arguments.begin())
            out << ", ";

          out << cpp_typecheck.to_string(argument_type);
        }

        if(code_type.has_ellipsis())
        {
          if(!arguments.empty())
            out << ", ";
          out << "...";
        }

        out << ")";
      }
      else
        out << id << ": " << cpp_typecheck.to_string(id_expr.type());

      if(id_expr.id() == "symbol")
      {
        const symbolt &symbol = cpp_typecheck.lookup(to_symbol_expr(id_expr));
        out << " (" << symbol.location << ")";
      }
      else if(id_expr.id() == "member")
      {
        const symbolt *symbol;
        bool found = cpp_typecheck.lookup(id_expr.component_name(), symbol);
        if(!found)
          out << " (" << symbol->location << ")";
      }
      else if(id_expr.id() == "template_function_instance")
      {
        const symbolt &symbol =
          cpp_typecheck.lookup(id_expr.type().get("#template"));
        out << " (" << symbol.location << ")";
      }
    }

    out << std::endl;
  }
}

exprt cpp_typecheck_resolvet::resolve(
  const cpp_namet &cpp_name,
  const wantt want,
  const cpp_typecheck_fargst &fargs,
  bool fail_with_exception)
{
  std::string base_name;
  cpp_template_args_non_tct template_args;
  template_args.make_nil();

  original_scope = &cpp_typecheck.cpp_scopes.current_scope();
  cpp_save_scopet save_scope(cpp_typecheck.cpp_scopes);

  // this changes the scope
  resolve_scope(cpp_name, base_name, template_args);

  const locationt &location = cpp_name.location();
  bool qualified = cpp_name.is_qualified();

  // do __CPROVER scope
  if(qualified)
  {
    if(cpp_typecheck.cpp_scopes.current_scope().identifier == "__CPROVER")
      return do_builtin(base_name, template_args);
  }
  else
  {
    if(base_name == "true")
    {
      exprt result;
      result.make_true();
      result.location() = location;
      return result;
    }
    if(base_name == "false")
    {
      exprt result;
      result.make_false();
      result.location() = location;
      return result;
    }
    else if(base_name == "__nullptr") // this is c++0x
    {
      constant_exprt result;
      result.set_value("NULL");
      result.type() = pointer_typet();
      result.type().subtype() = empty_typet();
      result.location() = location;
      return result;
    }
    else if(
      base_name == "__func__" || base_name == "__FUNCTION__" ||
      base_name == "__PRETTY_FUNCTION__")
    {
      // __func__ is an ANSI-C standard compliant hack to get the function name
      // __FUNCTION__ and __PRETTY_FUNCTION__ are GCC-specific
      string_constantt s(location.get_function());
      s.location() = location;
      return s;
    }
  }

  cpp_scopest::id_sett id_set;
  if(template_args.is_nil())
    cpp_typecheck.cpp_scopes.get_ids(base_name, id_set, false);
  else
    cpp_typecheck.cpp_scopes.get_ids(
      base_name, cpp_idt::TEMPLATE, id_set, false);

  // Argument-dependent name lookup
  if(!qualified && !fargs.has_object)
    resolve_with_arguments(id_set, base_name, fargs);

  if(id_set.empty())
  {
    if(!fail_with_exception)
      return nil_exprt();

    cpp_typecheck.show_instantiation_stack(cpp_typecheck.str);
    cpp_typecheck.err_location(location);
    cpp_typecheck.str << "symbol `" << base_name << "' not found";

    if(qualified)
    {
      if(cpp_typecheck.cpp_scopes.current_scope().is_root_scope())
        cpp_typecheck.str << " in root scope";
      else
        cpp_typecheck.str << " in scope `"
                          << cpp_typecheck.cpp_scopes.current_scope().prefix
                          << "'";
    }

    //cpp_typecheck.cpp_scopes.get_root_scope().print(std::cout);
    cpp_typecheck.cpp_scopes.current_scope().print(std::cout);
    throw 0;
  }

  resolve_identifierst identifiers;

  if(template_args.is_not_nil())
  {
    // first figure out if we are doing functions/methods or
    // classes
    bool have_classes = false, have_methods = false;

    for(auto it : id_set)
    {
      const irep_idt id = it->identifier;
      const symbolt &s = cpp_typecheck.lookup(id);
      assert(s.type.get_bool("is_template"));
      if(to_cpp_declaration(s.type).is_class_template())
        have_classes = true;
      else
        have_methods = true;
    }

    if(want == BOTH && have_classes && have_methods)
    {
      if(!fail_with_exception)
        return nil_exprt();

      cpp_typecheck.show_instantiation_stack(cpp_typecheck.str);
      cpp_typecheck.err_location(location);
      cpp_typecheck.str << "template symbol `" << base_name << "' is ambiguous";
      throw 0;
    }

    // We're classes
    if(want == TYPE || have_classes)
    {
      // Look for class specialization
      bool spec = false;
      for(auto it : id_set)
      {
        const irep_idt id = it->identifier;

        const symbolt &s = cpp_typecheck.lookup(id);
        if(!s.type.get_bool("is_template"))
          continue;

        const cpp_declarationt &cpp_declaration = to_cpp_declaration(s.type);
        if(!cpp_declaration.is_class_template())
          continue;

        irep_idt specialization_of = cpp_declaration.get_specialization_of();
        if(specialization_of != "")
          spec = true;
      }

      // Class specialization
      if(spec)
      {
        const symbolt &instance =
          disambiguate_template_classes(base_name, id_set, template_args);
        identifiers.emplace_back("type", symbol_typet(instance.name));
      }
    }

    if(!identifiers.size())
    {
      filter(id_set, want);

      convert_identifiers(id_set, want, fargs, identifiers);

      apply_template_args(identifiers, template_args, fargs);
    }
  }
  else
  {
    convert_identifiers(id_set, want, fargs, identifiers);
  }

  // change types into constructors if we want a constructor
  if(want == VAR)
    make_constructors(identifiers);

  if(identifiers.size() != 1)
    filter(identifiers, want);

  exprt result;

  // We disambiguate functions
  resolve_identifierst new_identifiers = identifiers;
  remove_templates(new_identifiers);

  // we only want _exact_ matches, without templates!
  exact_match_functions(new_identifiers, fargs);

  // no exact matches? Try again with function template guessing.
  if(new_identifiers.empty())
  {
    new_identifiers = identifiers;

    if(template_args.is_nil())
    {
      guess_function_template_args(new_identifiers, fargs);
    }
  }

  disambiguate_functions(new_identifiers, fargs);

  remove_duplicates(new_identifiers);

  disambiguate_copy_constructor(new_identifiers);

  if(new_identifiers.size() == 1)
  {
    result = *new_identifiers.begin();
  }
  else
  {
    // nothing or too many
    if(!fail_with_exception)
      return nil_exprt();

    if(new_identifiers.empty())
    {
      cpp_typecheck.err_location(location);
      cpp_typecheck.str << "found no match for symbol `" << base_name
                        << "', candidates are:" << std::endl;
      show_identifiers(base_name, identifiers, cpp_typecheck.str);
    }
    else
    {
      cpp_typecheck.err_location(location);
      cpp_typecheck.str << "symbol `" << base_name
                        << "' does not uniquely resolve:" << std::endl;
      show_identifiers(base_name, new_identifiers, cpp_typecheck.str);
    }

    if(fargs.in_use)
    {
      cpp_typecheck.str << std::endl;
      cpp_typecheck.str << "argument types:" << std::endl;

      for(const auto &operand : fargs.operands)
      {
        cpp_typecheck.str << "  " << cpp_typecheck.to_string(operand.type())
                          << std::endl;
      }
    }

    if(!cpp_typecheck.instantiation_stack.empty())
    {
      cpp_typecheck.str << std::endl;
      cpp_typecheck.show_instantiation_stack(cpp_typecheck.str);
    }

    throw 0;
  }

  switch(want)
  {
  case VAR:
    if(result.id() == "type" && !cpp_typecheck.cpp_is_pod(result.type()))
    {
      if(!fail_with_exception)
        return nil_exprt();

      cpp_typecheck.err_location(location);

      cpp_typecheck.str << "error: expected expression, but got type `"
                        << cpp_typecheck.to_string(result.type()) << "'";

      throw 0;
    }
    break;

  case TYPE:
    if(result.id() != "type")
    {
      if(!fail_with_exception)
        return nil_exprt();

      cpp_typecheck.err_location(location);

      cpp_typecheck.str << "error: expected type, but got expression `"
                        << cpp_typecheck.to_string(result) << "'";

      throw 0;
    }
    break;

  default:
    break;
  }

  // Workaround for SFINAE: mark any specuatively instantiated template that
  // we've resolved to, as in use. This means that any error in it will still
  // be reported.
  // XXX - this isn't going to work recursively.
  if(want == VAR && result.id() == "symbol")
  {
    symbolt &sym = *cpp_typecheck.context.find_symbol(result.identifier());
    if(sym.value.get("#speculative_template") == "1")
      sym.value.set("#template_in_use", "1");
  }
  else if(want == VAR && result.id() == "member")
  {
    // Is this a fake-member, i.e. a member with component_name == symbol name?
    symbolt *s = cpp_typecheck.context.find_symbol(result.component_name());
    if(s != nullptr)
    {
      if(s->value.get("#speculative_template") == "1")
        s->value.set("#template_in_use", "1");
    }
  }

  check_incomplete_template_class(result, want);

  return result;
}

void cpp_typecheck_resolvet::check_incomplete_template_class(
  exprt result,
  wantt want)
{
  // Check if the type is complete. The template might have been forward
  // declared so ESBMC didn't instantiated it.
  if(want != TYPE)
    return;

  const typet &t = result.type();

  if(t.identifier() == irep_idt())
    return;

  symbolt *s = cpp_typecheck.context.find_symbol(t.identifier());
  if(s != nullptr)
  {
    // It is a template and it wasn't instantiated?
    if(
      s->type.id() == "incomplete_struct" &&
      s->type.find("#template").is_not_nil())
    {
      exprt template_expr =
        static_cast<const exprt &>(s->type.find("#template"));

      symbolt *template_symbol =
        cpp_typecheck.context.find_symbol(template_expr.id());
      if(template_symbol != nullptr)
      {
        // If it is nil, it was forward declared, when it got a body,
        // we'll instantiate it
        if(template_symbol->type.type().body().is_not_nil())
        {
          cpp_template_args_tct instantiated_args =
            to_cpp_template_args_tc(s->type.find("#template_arguments"));

          cpp_typecheck.instantiate_template(
            location, *template_symbol, instantiated_args, instantiated_args);
        }
      }
    }
  }
}

void cpp_typecheck_resolvet::guess_template_args(
  const exprt &template_expr,
  const exprt &desired_expr)
{
  if(template_expr.id() == "cpp-name")
  {
    const cpp_namet &cpp_name = to_cpp_name(template_expr);

    if(!cpp_name.is_qualified())
    {
      cpp_save_scopet save_scope(cpp_typecheck.cpp_scopes);

      cpp_template_args_non_tct template_args;
      std::string base_name;
      resolve_scope(cpp_name, base_name, template_args);

      cpp_scopest::id_sett id_set;
      cpp_typecheck.cpp_scopes.get_ids(base_name, id_set, false);

      // alright, rummage through these
      for(auto it : id_set)
      {
        const cpp_idt &id = *it;
        // template argument?
        if(id.id_class == cpp_idt::TEMPLATE_ARGUMENT)
        {
          // see if unassigned
          exprt &e = cpp_typecheck.template_map.expr_map[id.identifier];
          if(e.id() == "unassigned")
          {
            typet old_type = e.type();
            e = desired_expr;
            if(e.type() != old_type)
              e.make_typecast(old_type);
          }
        }
      }
    }
  }
}

bool cpp_typecheck_resolvet::is_conversion_type_exact_match(
  const typet &source_type,
  const typet &dest_type)
{
  // Simplfy ask the implicit conversion sequence code. Construct a dummy symbol
  // first though for it to manipulate.
  symbol_exprt temp_src_value("fake_sym_for_templ_deduction_test", source_type);
  exprt output_expr;

  cpp_typecast_rank rank;
  if(!cpp_typecheck.implicit_conversion_sequence(
       temp_src_value, dest_type, output_expr, rank))
    return false; // No conversion.

  // Only permit argument deduction where there's an exact match. Otherwise,
  // this inteferes with overloading.
  if(rank.rank >= 2)
    return false;

  return true;
}

bool cpp_typecheck_resolvet::guess_template_args(
  const typet &template_type,
  const typet &desired_type)
{
  // look at
  // http://publib.boulder.ibm.com/infocenter/comphelp/v8v101/topic/com.ibm.xlcpp8a.doc/language/ref/template_argument_deduction.htm

  // T
  // const T
  // volatile T
  // T&
  // T*
  // T[10]
  // A<T>
  // C(*)(T)
  // T(*)()
  // T(*)(U)
  // T C::*
  // C T::*
  // T U::*
  // T (C::*)()
  // C (T::*)()
  // D (C::*)(T)
  // C (T::*)(U)
  // T (C::*)(U)
  // T (U::*)()
  // T (U::*)(V)
  // E[10][i]
  // B<i>
  // TT<T>
  // TT<i>
  // TT<C>

  //new stuff
  // const *T

  if(template_type.id() == "cpp-name")
  {
    // we only care about cpp_names that are template parameters!
    const cpp_namet &cpp_name = to_cpp_name(template_type);

    cpp_save_scopet save_scope(cpp_typecheck.cpp_scopes);

    if(cpp_name.has_template_args())
    {
      // This is a non-trivial name that has template arguments, for example
      // std::list<X>. We're expected to deduce the type of X from the argument.
      // To do that, a) the argument has to have instantiated template args
      // itself, b) it has to match the format (std::list, etc) of the
      // parameter, and c) not conflict with existing template arg assignments.

      // Start off by trying to look up whether the argument has template args
      // itself. No arguments means that substituation has failed.
      std::string base_name;
      cpp_template_args_non_tct template_args;
      resolve_scope(cpp_name, base_name, template_args);
      bool has_args = false;

      cpp_template_args_non_tct instantiated_args;
      if(desired_type.find("#cpp_type").is_not_nil())
      {
        type_exprt type(desired_type);
        instantiated_args.arguments().push_back(type);
        has_args = true;
      }
      else
      {
        // Does the argument type we're dealing with already have assigned
        // template arguments in its type?
        symbolt &s = const_cast<symbolt &>(
          cpp_typecheck.lookup(desired_type.identifier()));
        exprt &template_arguments = const_cast<exprt &>(
          static_cast<const exprt &>(s.type.find("#template_arguments")));

        if(template_arguments.is_not_nil())
        {
          instantiated_args = to_cpp_template_args_non_tc(template_arguments);
          has_args = true;
        }
        else
        {
          // No; we must look if the parent scope has the template arguments
          cpp_scopet &scope = cpp_typecheck.cpp_scopes.get_scope(s.name);

          unsigned parent_size = scope.parents_size();
          while(parent_size)
          {
            cpp_scopet &parent = scope.get_parent();

            // If we've reached the root scope, then we've failed to find
            // template arguments
            if(parent.id_class == cpp_idt::ROOT_SCOPE)
              break;

            const symbolt &s2 = cpp_typecheck.lookup(parent.identifier);

            const exprt &template_arguments2 =
              static_cast<const exprt &>(s2.type.find("#template_arguments"));

            if(template_arguments2.is_not_nil())
            {
              instantiated_args =
                to_cpp_template_args_non_tc(template_arguments2);
              has_args = true;
              break;
            }

            parent_size = parent.parents_size();
          }
        }
      }

      if(!has_args)
        // We were unable to find any template arguments in the argument type.
        // This means it definitely doesn't match this template, so substitution
        // has failed. Record this by leaving template args unassigned.
        // XXX: this is not robust. If another function argument assigns those
        // template arguments correctly, this template will be accepted. Better
        // error reporting is necessary.
        return false; // return value unused.

      cpp_template_args_non_tct::argumentst args = template_args.arguments();
      for(unsigned i = 0; i < args.size(); ++i)
      {
        cpp_namet cpp_name;
        if(args[i].id() == "unary-")
          cpp_name = to_cpp_name(args[i].op0());
        else
          cpp_name = to_cpp_name(args[i].type());

        resolve_scope(cpp_name, base_name, template_args);

        cpp_scopest::id_sett id_set;
        cpp_typecheck.cpp_scopes.get_ids(base_name, id_set, false);

        // alright, rummage through these
        for(auto it : id_set)
        {
          const cpp_idt &id = *it;

          // template argument?
          if(id.id_class == cpp_idt::TEMPLATE_ARGUMENT)
          {
            // see if unassigned
            typet &t = cpp_typecheck.template_map.type_map[id.identifier];

            if(t.id() == "unassigned")
            {
              assert(instantiated_args.arguments().size() > i);
              t = instantiated_args.arguments()[i].type();

              // remove const, volatile (these can be added in the call)
              t.remove("#constant");
              t.remove("#volatile");
            }
            else
            {
              // This template has been deduced from another argument. If they
              // don't have exactly matching types, then reject this deduction
              // (See spec 14.8.2.4.2).
              if(!is_conversion_type_exact_match(desired_type, t))
                return true;
            }
          }
        }
      }
    }
    else
    {
      // template parameters aren't qualified
      if(!cpp_name.is_qualified())
      {
        std::string base_name;
        cpp_template_args_non_tct template_args;
        resolve_scope(cpp_name, base_name, template_args);

        cpp_scopest::id_sett id_set;
        cpp_typecheck.cpp_scopes.get_ids(base_name, id_set, false);

        // alright, rummage through these
        for(auto it : id_set)
        {
          const cpp_idt &id = *it;

          // template argument?
          if(id.id_class == cpp_idt::TEMPLATE_ARGUMENT)
          {
            // see if unassigned
            typet &t = cpp_typecheck.template_map.type_map[id.identifier];

            if(t.id() == "unassigned")
            {
              t = desired_type;

              // remove const, volatile (these can be added in the call)
              t.remove("#constant");
              t.remove("#volatile");

              // XXX: various implicit conversions are defined by section 14.1
              // of the spec, most notably that array types become element
              // pointers. This should be refactored into a separate method
              // and fully explored, rather than randomly patched here.
              if(t.id() == "array")
              {
                // Morph irep.
                t.id("pointer");
                t.remove("size");
              }
            }
            else
            {
              // This template has been deduced from another argument. If they
              // don't have exactly matching types, then reject this deduction
              // (See spec 14.8.2.4.2).
              if(!is_conversion_type_exact_match(desired_type, t))
                return true;
            }
          }
        }
      }
    }
  }
  else if(template_type.id() == "merged_type")
  {
    // look at subtypes
    for(const auto &it : template_type.subtypes())
    {
      guess_template_args(it, desired_type);
    }
  }
  else if(is_reference(template_type) || is_rvalue_reference(template_type))
  {
    guess_template_args(template_type.subtype(), desired_type);
  }
  else if(template_type.id() == "pointer")
  {
    const typet &desired_type_followed = cpp_typecheck.follow(desired_type);

    if(
      desired_type_followed.id() == "pointer" ||
      desired_type_followed.id() == "array")
      guess_template_args(
        template_type.subtype(), desired_type_followed.subtype());
  }
  else if(template_type.id() == "array")
  {
    const typet &desired_type_followed = cpp_typecheck.follow(desired_type);

    if(desired_type_followed.id() == "array")
    {
      // look at subtype first
      guess_template_args(
        template_type.subtype(), desired_type_followed.subtype());

      // size (e.g., buffer size guessing)
      guess_template_args(
        to_array_type(template_type).size(),
        to_array_type(desired_type_followed).size());
    }
  }

  return false;
}

exprt cpp_typecheck_resolvet::guess_function_template_args(
  const exprt &expr,
  const cpp_typecheck_fargst &fargs)
{
  typet tmp = expr.type();
  cpp_typecheck.follow_symbol(tmp);

  // XXX -- Spec allows for partial template argument specification, currently
  // not handled.

  if(!tmp.get_bool("is_template"))
    return nil_exprt(); // not a template

  assert(expr.id() == "symbol");

  // a template is always a declaration
  const cpp_declarationt &cpp_declaration = to_cpp_declaration(tmp);

  // Class templates require explicit template arguments,
  // no guessing!
  if(cpp_declaration.is_class_template())
    return nil_exprt();

  // we need function arguments for guessing
  if(fargs.operands.empty())
    return nil_exprt(); // give up

  // We need to guess in the case of function templates!

  irep_idt template_identifier = to_symbol_expr(expr).get_identifier();

  const symbolt &template_symbol = cpp_typecheck.lookup(template_identifier);

  // alright, set up template arguments as 'unassigned'

  cpp_saved_template_mapt saved_map(cpp_typecheck.template_map);

  cpp_typecheck.template_map.build_unassigned(cpp_declaration.template_type());

  // there should be exactly one declarator
  assert(cpp_declaration.declarators().size() == 1);

  const cpp_declaratort &function_declarator =
    cpp_declaration.declarators().front();

  // and that needs to have function type
  if(function_declarator.type().id() != "function_type")
  {
    cpp_typecheck.err_location(location);
    throw "expected function type for function template";
  }

  cpp_save_scopet cpp_saved_scope(cpp_typecheck.cpp_scopes);

  // we need the template scope
  cpp_scopet *template_scope = static_cast<cpp_scopet *>(
    cpp_typecheck.cpp_scopes.id_map[template_identifier]);

  if(template_scope == nullptr)
  {
    cpp_typecheck.err_location(location);
    cpp_typecheck.str << "template identifier: " << template_identifier
                      << std::endl;
    throw "function template instantiation error";
  }

  // enter the scope of the template
  cpp_typecheck.cpp_scopes.go_to(*template_scope);

  // walk through the function arguments
  const irept::subt &arguments =
    function_declarator.type().arguments().get_sub();

  unsigned int i = 0, j = 0;
  // Method templates don't have the implicit 'this' pointer attached to them,
  // because they're not actually members of the class until they're
  // instantiated. If this template is being called in the context of an object
  // method invocation, skip the first argument.
  // XXX -- what about static methods? IIRC they have the implicit arg, but
  // it's ignored/
  if(fargs.has_object)
    i = 1;

  // In this context, j is used to indicate the arguments (the existing template
  // arguments), and i the operands, which are in the fargs object, and have
  // concrete types.
  for(; j < arguments.size(); i++, j++)
  {
    if(i < fargs.operands.size() && arguments[j].id() == "cpp-declaration")
    {
      const cpp_declarationt &arg_declaration =
        to_cpp_declaration(arguments[j]);

      // again, there should be one declarator
      assert(arg_declaration.declarators().size() == 1);

      // turn into type
      typet arg_type = arg_declaration.declarators().front().merge_type(
        arg_declaration.type());

      // We only convert the arg_type,
      // and don't typecheck it -- that could cause all
      // sorts of trouble.
      cpp_convert_plain_type(arg_type);

      // Guess the template argument; if there's an incompatibility, then
      // deduction has failed.
      if(guess_template_args(arg_type, fargs.operands[i].type()))
        return nil_exprt();
    }
  }

  // see if that has worked out
  cpp_template_args_tct template_args =
    cpp_typecheck.template_map.build_template_args(
      cpp_declaration.template_type());

  if(template_args.has_unassigned())
    return nil_exprt(); // give up

  // Build the type of the function.

  typet function_type = function_declarator.merge_type(cpp_declaration.type());

  cpp_typecheck.typecheck_type(function_type);

  // Remember that this was a template

  function_type.set("#template", template_symbol.name);
  function_type.set("#template_arguments", template_args);

  // Seems we got an instance for all parameters. Let's return that.

  exprt template_function_instance("template_function_instance", function_type);

  return template_function_instance;
}

void cpp_typecheck_resolvet::apply_template_args(
  exprt &expr,
  const cpp_template_args_non_tct &template_args_non_tc,
  const cpp_typecheck_fargst &fargs)
{
  if(expr.id() != "symbol")
    return; // templates are always symbols

  const symbolt &template_symbol = cpp_typecheck.lookup(expr.identifier());

  if(!template_symbol.type.get_bool("is_template"))
    return;

#if 0
  if(template_args_non_tc.is_nil())
  {
    // no arguments, need to guess
    guess_function_template_args(expr, fargs);
    return;
  }
#endif

  // We typecheck the template arguments in the context
  // of the original scope!
  cpp_template_args_tct template_args_tc;

  {
    cpp_save_scopet save_scope(cpp_typecheck.cpp_scopes);

    cpp_typecheck.cpp_scopes.go_to(*original_scope);

    template_args_tc = cpp_typecheck.typecheck_template_args(
      location, template_symbol, template_args_non_tc);
    // go back to where we used to be
  }

  // We never try 'unassigned' template arguments.
  if(template_args_tc.has_unassigned())
    assert(false);

  // a template is always a declaration
  const cpp_declarationt &cpp_declaration =
    to_cpp_declaration(template_symbol.type);

  // is it a class template or function template?
  if(cpp_declaration.is_class_template())
  {
    const symbolt &new_symbol = cpp_typecheck.instantiate_template(
      location, template_symbol, template_args_tc, template_args_tc);

    expr = exprt("type", symbol_typet(new_symbol.name));
    expr.location() = location;
  }
  else
  {
    // must be a function, maybe method
    const symbolt &new_symbol = cpp_typecheck.instantiate_template(
      location, template_symbol, template_args_tc, template_args_tc);

    // check if it is a method
    const code_typet &code_type = to_code_type(new_symbol.type);

    if(
      !code_type.arguments().empty() &&
      code_type.arguments()[0].cmt_base_name() == "this")
    {
      symbolt type_symb;

      // do we have an object?
      if(fargs.has_object)
        type_symb =
          cpp_typecheck.lookup(fargs.operands.begin()->type().identifier());
      else
        type_symb = cpp_typecheck.lookup(original_scope->class_identifier);

      assert(type_symb.type.id() == "struct");

#ifndef NDEBUG
      const struct_typet &struct_type = to_struct_type(type_symb.type);
      assert(struct_type.has_component(new_symbol.name));
#endif
      member_exprt member(code_type);
      member.set_component_name(new_symbol.name);
      if(fargs.has_object)
        member.struct_op() = *fargs.operands.begin();
      else
      {
        // Add (this) as argument of the function
        const exprt &this_expr = original_scope->this_expr;

        // use this->...
        assert(this_expr.type().id() == "pointer");

        exprt object = exprt("dereference", this_expr.type().subtype());
        object.copy_to_operands(this_expr);
        object.type().set(
          "#constant", this_expr.type().subtype().cmt_constant());
        object.set("#lvalue", true);
        object.location() = location;

        member.struct_op() = object;
      }
      member.location() = location;
      expr.swap(member);
      return;
    }

    expr = cpp_symbol_expr(new_symbol);
    expr.location() = location;
  }
}

bool cpp_typecheck_resolvet::disambiguate_functions(
  const exprt &expr,
  cpp_typecast_rank &args_distance,
  const cpp_typecheck_fargst &fargs)
{
  args_distance = cpp_typecast_rank();

  // Not code, not a function, bail.
  if(expr.type().id() != "code")
    return false;

  // No arguments -> nothing to disambiguate?
  if(!fargs.in_use)
    return true;

  const code_typet &type = to_code_type(expr.type());

  if(expr.id() == "member" || type.return_type().id() == "constructor")
  {
    // if it's a member, but does not have an object yet,
    // we add one
    if(!fargs.has_object)
    {
      const code_typet::argumentst &arguments = type.arguments();
      const code_typet::argumentt &argument = arguments.front();

      assert(argument.cmt_base_name() == "this");

      if(expr.type().get("return_type") == "constructor")
      {
        // it's a constructor
        const typet &object_type = argument.type().subtype();
        exprt object("symbol", object_type);
        object.set("#lvalue", true);

        cpp_typecheck_fargst new_fargs(fargs);
        new_fargs.add_object(object);
        return new_fargs.match(type, args_distance, cpp_typecheck);
      }

      if(
        expr.type().get_bool("#is_operator") &&
        fargs.operands.size() == arguments.size())
      {
        return fargs.match(type, args_distance, cpp_typecheck);
      }

      cpp_typecheck_fargst new_fargs(fargs);
      new_fargs.add_object(expr.op0());

      return new_fargs.match(type, args_distance, cpp_typecheck);
    }
  }
  else if(fargs.has_object)
  {
    // if it's not a member then we shall remove the object
    // jmorse: Actually, member function templates end up here, so we shouldn't.
    // Plus, normal function templates shouldn't be matched in member
    // invocations, which I imagine is what this would match as a result.

    //cpp_typecheck_fargst new_fargs(fargs);
    //new_fargs.remove_object();

    return fargs.match(type, args_distance, cpp_typecheck);
  }

  return fargs.match(type, args_distance, cpp_typecheck);
}

void cpp_typecheck_resolvet::filter_for_named_scopes(
  cpp_scopest::id_sett &id_set)
{
  cpp_scopest::id_sett new_set;

  // We only want scopes!
  for(auto it : id_set)
  {
    cpp_idt &id = *it;

    if(id.is_class() || id.is_enum() || id.is_namespace())
    {
      assert(id.is_scope);
      new_set.insert(&id);
    }
    else if(id.is_typedef())
    {
      irep_idt identifier = id.identifier;

      if(id.is_member)
      {
        struct_typet struct_type = static_cast<const struct_typet &>(
          cpp_typecheck.lookup(id.class_identifier).type);
        const exprt pcomp = struct_type.get_component(identifier);
        assert(pcomp.is_not_nil());
        assert(pcomp.is_type());
        const typet &type = pcomp.type();
        assert(type.id() != "struct");
        if(type.id() == "symbol")
          identifier = type.identifier();
        else
          continue;
      }

      while(true)
      {
        const symbolt &symbol = cpp_typecheck.lookup(identifier);
        assert(symbol.is_type);

        // todo? maybe do enum here, too?
        if(symbol.type.id() == "struct")
        {
          // this is a scope, too!
          cpp_idt &class_id = cpp_typecheck.cpp_scopes.get_id(identifier);

          assert(class_id.is_scope);
          new_set.insert(&class_id);
          break;
        }
        if(symbol.type.id() == "symbol")
          identifier = symbol.type.identifier();
        else
          break;
      }
    }
    else if(id.id_class == cpp_scopet::TEMPLATE)
    {
      const symbolt symbol = cpp_typecheck.lookup(id.identifier);
      if(symbol.type.get("type") == "struct")
        new_set.insert(&id);
    }
    else if(id.id_class == cpp_scopet::TEMPLATE_ARGUMENT)
    {
      // a template argument may be a scope: it could
      // be instantiated with a class/struct/union/enum
      exprt e = convert_template_argument(id);

      if(e.id() != "type")
        continue; // expressions are definitively not a scope

      if(e.type().id() == "symbol")
      {
        symbol_typet type = to_symbol_type(e.type());

        while(true)
        {
          irep_idt identifier = type.identifier();

          const symbolt &symbol = cpp_typecheck.lookup(identifier);
          assert(symbol.is_type);

          if(symbol.type.id() == "symbol")
            type = to_symbol_type(symbol.type);
          if(
            symbol.type.id() == "struct" ||
            symbol.type.id() == "incomplete_struct" ||
            symbol.type.id() == "union" ||
            symbol.type.id() == "incomplete_union" ||
            symbol.type.id() == "c_enum")
          {
            // this is a scope, too!
            cpp_idt &class_id = cpp_typecheck.cpp_scopes.get_id(identifier);

            assert(class_id.is_scope);
            new_set.insert(&class_id);
            break;
          }
          // give up
          break;
        }
      }
    }
  }

  id_set.swap(new_set);
}

void cpp_typecheck_resolvet::filter_for_namespaces(cpp_scopest::id_sett &id_set)
{
  // we only want namespaces
  for(cpp_scopest::id_sett::iterator it = id_set.begin();
      it != id_set.end();) // no it++
  {
    if((*it)->is_namespace())
      it++;
    else
    {
      cpp_scopest::id_sett::iterator old(it);
      it++;
      id_set.erase(old);
    }
  }
}

void cpp_typecheck_resolvet::resolve_with_arguments(
  cpp_scopest::id_sett &id_set,
  const std::string &base_name,
  const cpp_typecheck_fargst &fargs)
{
  cpp_save_scopet save_scope(cpp_typecheck.cpp_scopes);

  for(const auto &operand : fargs.operands)
  {
    const typet &final_type = cpp_typecheck.follow(operand.type());

    if(final_type.id() != "struct" && final_type.id() != "union")
      continue;

    cpp_scopest::id_sett tmp_set;
    cpp_idt &scope = cpp_typecheck.cpp_scopes.get_scope(final_type.name());
    cpp_typecheck.cpp_scopes.go_to(scope);
    cpp_typecheck.cpp_scopes.get_ids(base_name, tmp_set, false);
    id_set.insert(tmp_set.begin(), tmp_set.end());
  }
}
