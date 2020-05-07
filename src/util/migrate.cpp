#include <util/c_types.h>
#include <util/config.h>
#include <util/irep2_utils.h>
#include <util/migrate.h>
#include <util/namespace.h>
#include <util/prefix.h>
#include <util/simplify_expr.h>
#include <util/type_byte_size.h>

// File for old irep -> new irep conversions.

// Due to the sins of the fathers, we need to have a namespace available during
// migration. There are now @ symbols embedded in variable names, so we can't
// detect the renaming level of a variable effectively. So, perform some hacks,
// by getting the top level parseoptions code to give us the global namespace,
// and use that to detect whether the symbol is renamed at all.
//
// Why is this a global? Because there are over three hundred call sites to
// migrate_expr, and it's a huge task to fix them all up to pass a namespace
// down.
namespacet *migrate_namespace_lookup = nullptr;

static std::map<irep_idt, BigInt> bin2int_map_signed, bin2int_map_unsigned;

const BigInt &binary2bigint(irep_idt binary, bool is_signed)
{
  std::map<irep_idt, BigInt> &ref =
    (is_signed) ? bin2int_map_signed : bin2int_map_unsigned;

  std::map<irep_idt, BigInt>::iterator it = ref.find(binary);
  if(it != ref.end())
    return it->second;
  BigInt val = binary2integer(binary.as_string(), is_signed);

  std::pair<std::map<irep_idt, BigInt>::const_iterator, bool> res =
    ref.insert(std::pair<irep_idt, BigInt>(binary, val));
  return res.first->second;
}

static expr2tc fixup_containerof_in_sizeof(const expr2tc &_expr)
{
  if(is_nil_expr(_expr))
    return _expr;

  expr2tc expr = _expr;

  // Blast through all typedefs
  while(is_typecast2t(expr))
    expr = to_typecast2t(expr).from;

  // Base must be null; must start with an addressof.
  if(!is_address_of2t(expr))
    return expr;

  const address_of2t &addrof = to_address_of2t(expr);
  return compute_pointer_offset(addrof.ptr_obj);
}

void real_migrate_type(
  const typet &type,
  type2tc &new_type_ref,
  const namespacet *ns,
  bool cache)
{
  if(type.id() == typet::t_bool)
  {
    bool_type2t *b = new bool_type2t();
    new_type_ref = type2tc(b);
  }
  else if(type.id() == typet::t_signedbv)
  {
    irep_idt width = type.width();
    unsigned int iwidth = strtol(width.as_string().c_str(), nullptr, 10);
    signedbv_type2t *s = new signedbv_type2t(iwidth);
    new_type_ref = type2tc(s);
  }
  else if(type.id() == typet::t_unsignedbv)
  {
    irep_idt width = type.width();
    unsigned int iwidth = strtol(width.as_string().c_str(), nullptr, 10);
    unsignedbv_type2t *s = new unsignedbv_type2t(iwidth);
    new_type_ref = type2tc(s);
  }
  else if(type.id() == "c_enum" || type.id() == "incomplete_c_enum")
  {
    // 6.7.2.2.3 of C99 says enumeration values shall have "int" types.
    signedbv_type2t *s = new signedbv_type2t(config.ansi_c.int_width);
    new_type_ref = type2tc(s);
  }
  else if(type.id() == typet::t_array)
  {
    type2tc subtype;
    expr2tc size((expr2t *)nullptr);
    bool is_infinite = false;

    migrate_type(type.subtype(), subtype, ns, cache);

    if(type.find(typet::a_size).id() == "infinity")
    {
      is_infinite = true;
    }
    else
    {
      exprt sz = (exprt &)type.find(typet::a_size);
      simplify(sz);
      migrate_expr(sz, size);
      size = fixup_containerof_in_sizeof(size);
    }

    array_type2t *a = new array_type2t(subtype, size, is_infinite);
    new_type_ref = type2tc(a);
  }
  else if(type.id() == typet::t_pointer)
  {
    type2tc subtype;

    // Don't recursively look up anything through pointers.
    migrate_type(type.subtype(), subtype, nullptr, true);

    pointer_type2t *p = new pointer_type2t(subtype);
    new_type_ref = type2tc(p);
  }
  else if(type.id() == typet::t_empty)
  {
    empty_type2t *e = new empty_type2t();
    new_type_ref = type2tc(e);
  }
  else if(type.id() == typet::t_symbol)
  {
    if(ns)
    {
      typet followed = ns->follow(type);
      real_migrate_type(followed, new_type_ref, ns, cache);
    }
    else
    {
      symbol_type2t *s = new symbol_type2t(type.identifier());
      new_type_ref = type2tc(s);
    }
  }
  else if(type.id() == typet::t_struct)
  {
    std::vector<type2tc> members;
    std::vector<irep_idt> names;
    std::vector<irep_idt> pretty_names;
    const struct_typet &strct = to_struct_type(type);
    const struct_union_typet::componentst comps = strct.components();

    for(const auto &comp : comps)
    {
      type2tc ref;

      // Hacks: due to SFINAE, we might have a templated instantiation in
      // here that's invalid. Try to detect those and avoid converting them.
      // As that's just going to cause grief.
      const irep_idt &this_name = comp.get("name");
      if(this_name != "" && ns)
      {
        const symbolt *component_sym;
        bool failed = ns->lookup(this_name, component_sym);
        if(
          !failed && component_sym->value.get("#speculative_template") == "1" &&
          component_sym->value.get("#template_in_use") != "1")
          // Skip it.
          continue;
      }

      migrate_type((const typet &)comp.type(), ref, ns, cache);

      members.push_back(ref);
      names.push_back(comp.get(typet::a_name));
      pretty_names.push_back(comp.get(typet::a_pretty_name));
    }

    irep_idt name = type.get("tag");
    if(name.as_string() == "")
      name = type.get("name"); // C++

    irep_idt ispacked = type.get("packed");
    bool packed = false;
    if(ispacked.as_string() == "true")
      packed = true;

    struct_type2t *s =
      new struct_type2t(members, names, pretty_names, name, packed);
    new_type_ref = type2tc(s);
  }
  else if(type.id() == typet::t_union)
  {
    std::vector<type2tc> members;
    std::vector<irep_idt> names;
    std::vector<irep_idt> pretty_names;
    const struct_union_typet &strct = to_union_type(type);
    const struct_union_typet::componentst comps = strct.components();

    for(const auto &comp : comps)
    {
      type2tc ref;
      migrate_type((const typet &)comp.type(), ref, ns, cache);

      members.push_back(ref);
      names.push_back(comp.get(typet::a_name));
      pretty_names.push_back(comp.get(typet::a_pretty_name));
    }

    irep_idt name = type.get("tag");
    assert(name.as_string() != "");
    union_type2t *u = new union_type2t(members, names, pretty_names, name);
    new_type_ref = type2tc(u);
  }
  else if(type.id() == typet::t_fixedbv)
  {
    unsigned int width_bits = to_fixedbv_type(type).get_width();
    unsigned int int_bits = to_fixedbv_type(type).get_integer_bits();

    fixedbv_type2t *f = new fixedbv_type2t(width_bits, int_bits);
    new_type_ref = type2tc(f);
  }
  else if(type.id() == typet::t_floatbv)
  {
    unsigned int frac_bits = to_floatbv_type(type).get_f();
    unsigned int expo_bits = to_floatbv_type(type).get_e();

    floatbv_type2t *f = new floatbv_type2t(frac_bits, expo_bits);
    new_type_ref = type2tc(f);
  }
  else if(type.id() == typet::t_code)
  {
    const code_typet &ref = static_cast<const code_typet &>(type);

    std::vector<type2tc> args;
    std::vector<irep_idt> arg_names;
    type2tc ret_type;
    bool ellipsis = false;

    if(ref.has_ellipsis())
      ellipsis = true;

    const code_typet::argumentst &old_args = ref.arguments();
    for(const auto &old_arg : old_args)
    {
      type2tc tmp;
      migrate_type(old_arg.type(), tmp, ns, cache);
      args.push_back(tmp);
      arg_names.push_back(old_arg.get_identifier());
    }

    // Don't migrate return type if it's a symbol. There are a variety of C++
    // things where a method returns itself, or similar.
    if(type.return_type().id() == "symbol")
    {
      ret_type = type2tc(new symbol_type2t(type.return_type().identifier()));
    }
    else
    {
      migrate_type(
        static_cast<const typet &>(type.return_type()), ret_type, ns, cache);
    }

    code_type2t *c = new code_type2t(args, ret_type, arg_names, ellipsis);
    new_type_ref = type2tc(c);
  }
  else if(type.id() == "cpp-name")
  {
    // No type,
    std::vector<type2tc> template_args;
    const exprt &cpy = (const exprt &)type;
    assert(cpy.get_sub()[0].id() == "name");
    irep_idt name = cpy.get_sub()[0].identifier();

    // Fetch possibly nonexistant template arguments.
    if(cpy.operands().size() == 2)
    {
      assert(cpy.get_sub()[0].id() == "template_args");
      forall_irep(it, cpy.get_sub())
      {
        assert((*it).id() == "type");
        type2tc tmptype;
        migrate_type((*it).type(), tmptype, ns, cache);
        template_args.push_back(tmptype);
      }
    }

    new_type_ref = type2tc(new cpp_name_type2t(name, template_args));
  }
  else if(type.id().as_string().size() == 0 || type.id() == "nil")
  {
    new_type_ref = type2tc(type_pool.get_empty());
  }
  else if(type.id() == "ellipsis")
  {
    // Eh? Ellipsis isn't a type. It's a special case.
    new_type_ref = type_pool.get_empty();
  }
  else if(type.id() == "destructor")
  {
    // This is a destructor return type. Which is nil.
    new_type_ref = type_pool.get_empty();
  }
  else if(type.id() == "constructor")
  {
    // New operator returns something; constructor is a void method on an
    // existing object.
    new_type_ref = type_pool.get_empty();
  }
  else if(type.id() == "incomplete_array")
  {
    // Hurrr. Mark as being infinite in size.
    // XXX find a way of ensuring that only extern-qualified arrays are handled
    // here, and nothing else can get here?
    type2tc subtype;
    expr2tc size;

    migrate_type(type.subtype(), subtype);

    array_type2t *a = new array_type2t(subtype, size, true);
    new_type_ref = type2tc(a);
  }
  else if(type.id() == "incomplete_struct")
  {
    // Only time that this occurs and the type checking code doesn't complain,
    // is when we take the /address/ of an incomplete struct. That's fine,
    // because we still can't access it as an incomplete struct. So just return
    // an infinitely sized array of characters, the most permissive approach to
    // something that shouldn't happen.
    new_type_ref = type2tc(new array_type2t(get_uint8_type(), expr2tc(), true));
  }
  else if(type.id() == "string")
  {
    irep_idt width = type.width();
    unsigned int iwidth = strtol(width.as_string().c_str(), nullptr, 10);
    new_type_ref = type2tc(new string_type2t(iwidth));
  }
  else
  {
    type.dump();
    assert(0);
  }
}

void migrate_type(
  const typet &type,
  type2tc &new_type_ref,
  const namespacet *ns,
  bool cache)
{
  if(!cache)
    return real_migrate_type(type, new_type_ref, ns, cache);

  if(type.id() == typet::t_bool)
  {
    new_type_ref = type_pool.get_bool();
  }
  else if(type.id() == typet::t_signedbv)
  {
    new_type_ref = type_pool.get_signedbv(type);
  }
  else if(type.id() == typet::t_unsignedbv)
  {
    new_type_ref = type_pool.get_unsignedbv(type);
  }
  else if(type.id() == "c_enum" || type.id() == "incomplete_c_enum")
  {
    // 6.7.2.2.3 of C99 says enumeration values shall have "int" types.
    new_type_ref = type_pool.get_int(config.ansi_c.int_width);
  }
  else if(type.id() == typet::t_array)
  {
    new_type_ref = type_pool.get_array(type);
  }
  else if(type.id() == typet::t_pointer)
  {
    new_type_ref = type_pool.get_pointer(type);
  }
  else if(type.id() == typet::t_empty)
  {
    new_type_ref = type_pool.get_empty();
  }
  else if(type.id() == typet::t_symbol)
  {
    if(ns)
    {
      typet followed = ns->follow(type);
      return migrate_type(followed, new_type_ref, ns, cache);
    }

    new_type_ref = type_pool.get_symbol(type);
  }
  else if(type.id() == typet::t_struct)
  {
    new_type_ref = type_pool.get_struct(type);
  }
  else if(type.id() == typet::t_union)
  {
    new_type_ref = type_pool.get_union(type);
  }
  else if(type.id() == typet::t_fixedbv)
  {
    new_type_ref = type_pool.get_fixedbv(type);
  }
  else if(type.id() == typet::t_floatbv)
  {
    new_type_ref = type_pool.get_floatbv(type);
  }
  else if(type.id() == typet::t_code)
  {
    new_type_ref = type_pool.get_code(type);
  }
  else if(type.id().as_string().size() == 0 || type.id() == "nil")
  {
    new_type_ref = type2tc(type_pool.get_empty());
  }
  else if(type.id() == "destructor")
  {
    // No such thing as returning a destructor; this appears to turn up when
    // there's a call to a destructor. Return type should be empty.
    new_type_ref = type_pool.get_empty();
  }
  else if(type.id() == "constructor")
  {
    // Likewise, a constructor returns nothing, the new operator returns
    // something.
    new_type_ref = type_pool.get_empty();
  }
  else if(type.id() == "cpp-name")
  {
    real_migrate_type(type, new_type_ref, ns, cache);
  }
  else if(type.id() == "ellipsis")
  {
    real_migrate_type(type, new_type_ref, ns, cache);
  }
  else if(type.id() == "incomplete_array")
  {
    real_migrate_type(type, new_type_ref, ns, cache);
  }
  else if(type.id() == "incomplete_struct")
  {
    real_migrate_type(type, new_type_ref, ns, cache);
  }
  else if(type.id() == "string")
  {
    real_migrate_type(type, new_type_ref, ns, cache);
  }
  else
  {
    type.dump();
    assert(0);
  }
}

static const typet &decide_on_expr_type(const exprt &side1, const exprt &side2)
{
  // For some arithmetic expr, decide on the result of operating on them.

  // First, if either are pointers, use that.
  if(side1.type().id() == typet::t_pointer)
    return side1.type();
  if(side2.type().id() == typet::t_pointer)
    return side2.type();

  // Then, fixedbv's/floatbv's take precedence.
  if(
    (side1.type().id() == typet::t_fixedbv) ||
    side1.type().id() == typet::t_floatbv)
    return side1.type();
  if(
    (side2.type().id() == typet::t_fixedbv) ||
    side2.type().id() == typet::t_floatbv)
    return side2.type();

  // If one operand is bool, return the other, as that's either bool or will
  // have a higher rank.
  if(side1.type().id() == typet::t_bool)
    return side2.type();
  if(side2.type().id() == typet::t_bool)
    return side1.type();

  assert(
    side1.type().id() == typet::t_unsignedbv ||
    side1.type().id() == typet::t_signedbv);
  assert(
    side2.type().id() == typet::t_unsignedbv ||
    side2.type().id() == typet::t_signedbv);

  unsigned int side1_width = atoi(side1.type().width().as_string().c_str());
  unsigned int side2_width = atoi(side2.type().width().as_string().c_str());

  if(side1.type().id() == side2.type().id())
  {
    if(side1_width > side2_width)
      return side1.type();

    return side2.type();
  }

  // Differing between signed/unsigned bv type. Take unsigned if greatest.
  if(side1.type().id() == typet::t_unsignedbv && side1_width >= side2_width)
    return side1.type();

  if(side2.type().id() == typet::t_unsignedbv && side2_width >= side1_width)
    return side2.type();

  // Otherwise return the signed one;
  if(side1.type().id() == typet::t_signedbv)
    return side1.type();

  return side2.type();
}

// Called when we have an expression (such as 'add') with more than two
// operands. irep2 only allows for binary expressions, so these have to be
// decomposed into a chain of add expressions, or similar.
static exprt splice_expr(const exprt &expr)
{
  // Duplicate
  exprt expr_recurse = expr;

  // Have we reached the bottom?
  if(expr.operands().size() == 2)
  {
    // Finish; optionally deduce type.
    if(expr.type().id() == "nil")
    {
      const typet &subexpr_type = decide_on_expr_type(expr.op0(), expr.op1());
      expr_recurse.type() = subexpr_type;
    }
    return expr_recurse;
  }

  // Remove back operand from recursive expr.
  exprt popped = expr_recurse.operands()[expr_recurse.operands().size() - 1];
  expr_recurse.operands().pop_back();

  // Set type to nil, so that subsequent calls to slice_expr deduce the
  // type themselves.
  expr_recurse.type().id("nil");
  exprt base = splice_expr(expr_recurse);

  // We now have an expr covering the rest of the expr, and an additional
  // operand; combine them into a new binary operation.
  exprt expr_twopart(expr.id());
  expr_twopart.copy_to_operands(base, popped);

  // Pick a type; if the incoming expr has no type, deduce it; if it does have
  // a type, use that one.
  if(expr.type().id() == "nil")
  {
    const typet &subexpr_type = decide_on_expr_type(base, popped);
    expr_twopart.type() = subexpr_type;
  }
  else
  {
    expr_twopart.type() = expr.type();
  }

  assert(expr_twopart.type().id() != "nil");
  return expr_twopart;
}

static void splice_expr(const exprt &expr, expr2tc &new_expr_ref)
{
  exprt newexpr = splice_expr(expr);
  migrate_expr(newexpr, new_expr_ref);
}

static void
convert_operand_pair(const exprt &expr, expr2tc &arg1, expr2tc &arg2)
{
  migrate_expr(expr.op0(), arg1);
  migrate_expr(expr.op1(), arg2);
}

expr2tc sym_name_to_symbol(irep_idt init, type2tc type)
{
  const symbolt *sym;
  symbol2t::renaming_level target_level;
  unsigned int level1_num = 0, thread_num = 0, node_num = 0, level2_num = 0;

  const std::string &thestr = init.as_string();
  // If this is an existing symbol name, then we're not renamed at all. Can't
  // rely on @ and ! symbols in the string "sadly".
  if(migrate_namespace_lookup->lookup(init, sym) == false)
  {
    // This is a level0 name.

    // Funkyness: use the global symbol table type. Why? Because various things
    // out there get parsed in with a partial type, i.e. something where a
    // function prototype is declared, or perhaps a pointer to an incomplete
    // type (but that's less of an issue). This then screws up future hash
    // tables, where symbols can have different types, and thus have different
    // hashes.
    // Fix this by ensuring that /all/ symbols with the same name use the type
    // from the global symbol table.
    migrate_type(sym->type, type);
    return expr2tc(new symbol2t(type, init, symbol2t::level0, 0, 0, 0, 0));
  }
  if(
    init.as_string().compare(0, 3, "cs$") == 0 ||
    init.as_string().compare(0, 8, "kindice$") == 0 ||
    init.as_string().compare(0, 2, "s$") == 0 ||
    init.as_string().compare(0, 5, "i$") == 0)
  {
    // This is part of k-induction, where the type is slowly accumulated over
    // time, and the symbol never makes its way into the symbol table :|
    return expr2tc(new symbol2t(type, init, symbol2t::level0, 0, 0, 0, 0));
  }

  // Renamed to at least level 1,
  size_t at_pos = thestr.rfind("?");
  size_t exm_pos = thestr.rfind("!");
  size_t end_of_name_pos = at_pos;

  size_t and_pos, hash_pos;
  if(thestr.find("#") == std::string::npos)
  {
    // We're level 1.
    target_level = symbol2t::level1;
    and_pos = thestr.size();
    hash_pos = thestr.size();
  }
  else
  {
    // Level 2
    target_level = symbol2t::level2;
    and_pos = thestr.find("&");
    hash_pos = thestr.find("#");

    if(at_pos == std::string::npos)
    {
      // However, it's L2 global.
      target_level = symbol2t::level2_global;
      end_of_name_pos = and_pos;
    }
  }

  // Whatever level we're at, set the base name to be nonrenamed.
  irep_idt thename = irep_idt(thestr.substr(0, end_of_name_pos));

  if(target_level != symbol2t::level2_global)
  {
    std::string atstr = thestr.substr(at_pos + 1, exm_pos - at_pos - 1);
    std::string exmstr = thestr.substr(exm_pos + 1, and_pos - exm_pos - 1);

    char *endatptr, *endexmptr;
    level1_num = strtol(atstr.c_str(), &endatptr, 10);
    assert(endatptr != atstr.c_str());
    thread_num = strtol(exmstr.c_str(), &endexmptr, 10);
    assert(endexmptr != exmstr.c_str());
  }

  if(target_level == symbol2t::level1)
  {
    return expr2tc(
      new symbol2t(type, thename, target_level, level1_num, 0, thread_num, 0));
  }

  std::string andstr = thestr.substr(and_pos + 1, hash_pos - and_pos - 1);
  std::string hashstr =
    thestr.substr(hash_pos + 1, thestr.size() - hash_pos - 1);

  node_num = atoi(andstr.c_str());
  level2_num = atoi(hashstr.c_str());
  return expr2tc(new symbol2t(
    type, thename, target_level, level1_num, level2_num, thread_num, node_num));
}

// Functions to flatten union literals to not contain anything of union type.
// Everything should become a byte array, as we slowly purge concrete unions
static expr2tc flatten_union(const exprt &expr);

static void flatten_to_bytes(const exprt &expr, std::vector<expr2tc> &bytes)
{
  // Migrate to irep2 for sanity. We can't have recursive unions.
  expr2tc new_expr;
  migrate_expr(expr, new_expr);

  // Awkwardly, this array literal might not be completely fixed-value, if
  // encoded in the middle of a function body or something that refers to other
  // variables.
  if(is_array_type(new_expr))
  {
    // Assume only fixed-size arrays (because you can't have variable size
    // members of unions).
    const array_type2t &arraytype = to_array_type(new_expr->type);
    assert(
      !arraytype.size_is_infinite && !is_nil_expr(arraytype.array_size) &&
      is_constant_int2t(arraytype.array_size) &&
      "Can't flatten array in union literal with unbounded size");

    // Iterate over each field and flatten to bytes
    const constant_int2t &intref = to_constant_int2t(arraytype.array_size);
    for(unsigned int i = 0; i < intref.value.to_uint64(); i++)
    {
      index2tc idx(arraytype.subtype, new_expr, gen_ulong(i));
      flatten_to_bytes(migrate_expr_back(idx), bytes);
    }
  }
  else if(is_struct_type(new_expr))
  {
    // Iterate over each field.
    const struct_type2t &structtype = to_struct_type(new_expr->type);
    for(unsigned long i = 0; i < structtype.members.size(); i++)
    {
      member2tc memb(
        structtype.members[i], new_expr, structtype.member_names[i]);
      flatten_to_bytes(migrate_expr_back(memb), bytes);
    }
  }
  else if(is_union_type(new_expr))
  {
    // This is an expression that evaluates to a union -- probably a symbol
    // name. It can't be a union literal, because that would have been
    // recursively flattened. In this circumstance we are *not* required to
    // actually perform any flattening, because something else in the union
    // transformation should have transformed it to a byte array. Simply take
    // the address (it has to have storage), cast to byte array, and index.
    BigInt size = type_byte_size(new_expr->type);
    address_of2tc addrof(new_expr->type, new_expr);
    type2tc byteptr(new pointer_type2t(get_uint8_type()));
    typecast2tc cast(byteptr, addrof);

    // Produce N bytes
    for(unsigned int i = 0; i < size.to_uint64(); i++)
    {
      index2tc idx(get_uint8_type(), cast, gen_ulong(i));
      flatten_to_bytes(migrate_expr_back(idx), bytes);
    }
  }
  else if(is_number_type(new_expr) || is_pointer_type(new_expr))
  {
    BigInt size = type_byte_size(new_expr->type);

    bool is_big_endian =
      config.ansi_c.endianess == configt::ansi_ct::IS_BIG_ENDIAN;
    for(unsigned int i = 0; i < size.to_uint64(); i++)
    {
      byte_extract2tc ext(
        get_uint8_type(), new_expr, gen_ulong(i), is_big_endian);
      bytes.push_back(ext);
    }
  }
  else
  {
    std::cerr << "Unrecognized type " << get_type_id(*new_expr->type);
    std::cerr << " when flattening union literal" << std::endl;
    abort();
  }
}

static expr2tc flatten_union(const exprt &expr)
{
  type2tc type;
  migrate_type(expr.type(), type);
  BigInt full_size = type_byte_size(type);

  // Union literals should have one field.
  assert(
    expr.operands().size() == 1 && "Union literal with more than one field");

  // Cannot have unbounded size; flatten to an array of bytes.
  std::vector<expr2tc> byte_array;
  flatten_to_bytes(expr.op0(), byte_array);

  // Potentially extend this array further if this literal is smaller than
  // the overall size of the union.
  expr2tc abyte = gen_zero(get_uint8_type());
  while(byte_array.size() < full_size.to_uint64())
    byte_array.push_back(abyte);

  expr2tc size = gen_ulong(byte_array.size());
  type2tc arraytype(new array_type2t(get_uint8_type(), size, false));
  constant_array2tc arr(arraytype, byte_array);
  return arr;
}

void migrate_expr(const exprt &expr, expr2tc &new_expr_ref)
{
  type2tc type;

  if(expr.id() == "nil")
  {
    new_expr_ref = expr2tc();
  }
  else if(expr.id() == irept::id_symbol)
  {
    migrate_type(expr.type(), type);
    new_expr_ref = sym_name_to_symbol(expr.identifier(), type);
  }
  else if(expr.id() == "nondet_symbol")
  {
    migrate_type(expr.type(), type);
    new_expr_ref = symbol2tc(type, "nondet$" + expr.identifier().as_string());
  }
  else if(
    expr.id() == irept::id_constant && expr.type().id() != typet::t_pointer &&
    expr.type().id() != typet::t_bool && expr.type().id() != "c_enum" &&
    expr.type().id() != typet::t_fixedbv &&
    expr.type().id() != typet::t_floatbv && expr.type().id() != typet::t_array)
  {
    migrate_type(expr.type(), type);

    bool is_signed = false;
    if(type->type_id == type2t::signedbv_id)
      is_signed = true;

    BigInt val = binary2bigint(expr.value(), is_signed);

    expr2t *new_expr = new constant_int2t(type, val);
    new_expr_ref = expr2tc(new_expr);
  }
  else if(expr.id() == irept::id_constant && expr.type().id() == "c_enum")
  {
    migrate_type(expr.type(), type);

    uint64_t enumval = atoi(expr.value().as_string().c_str());

    expr2t *new_expr = new constant_int2t(type, BigInt(enumval));
    new_expr_ref = expr2tc(new_expr);
  }
  else if(expr.id() == irept::id_constant && expr.type().id() == typet::t_bool)
  {
    std::string theval = expr.value().as_string();
    if(theval == "true")
      new_expr_ref = gen_true_expr();
    else
      new_expr_ref = gen_false_expr();
  }
  else if(
    expr.id() == irept::id_constant && expr.type().id() == typet::t_pointer &&
    expr.value() == "NULL")
  {
    // Null is a symbol with pointer type.
    migrate_type(expr.type(), type);

    expr2t *new_expr = new symbol2t(type, std::string("NULL"));
    new_expr_ref = expr2tc(new_expr);
  }
  else if(
    expr.id() == irept::id_constant && expr.type().id() == typet::t_fixedbv)
  {
    migrate_type(expr.type(), type);

    fixedbvt bv(to_constant_expr(expr));

    expr2t *new_expr = new constant_fixedbv2t(bv);
    new_expr_ref = expr2tc(new_expr);
  }
  else if(
    expr.id() == irept::id_constant && expr.type().id() == typet::t_floatbv)
  {
    migrate_type(expr.type(), type);

    ieee_floatt bv(to_constant_expr(expr));

    expr2t *new_expr = new constant_floatbv2t(bv);
    new_expr_ref = expr2tc(new_expr);
  }
  else if(expr.id() == exprt::typecast)
  {
    assert(expr.op0().id_string() != "");
    migrate_type(expr.type(), type);

    expr2tc old_expr;
    migrate_expr(expr.op0(), old_expr);

    // Default to rounding mode symbol
    expr2tc rounding_mode =
      expr2tc(new symbol2t(type_pool.get_int32(), "c:@__ESBMC_rounding_mode"));

    // If it's not nil, convert it
    exprt old_rm = expr.find_expr("rounding_mode");
    if(old_rm.is_not_nil())
      migrate_expr(old_rm, rounding_mode);

    typecast2t *t = new typecast2t(type, old_expr, rounding_mode);
    new_expr_ref = expr2tc(t);
  }
  else if(expr.id() == "bitcast")
  {
    assert(expr.op0().id_string() != "");
    migrate_type(expr.type(), type);

    expr2tc old_expr;
    migrate_expr(expr.op0(), old_expr);

    bitcast2t *t = new bitcast2t(type, old_expr);
    new_expr_ref = expr2tc(t);
  }
  else if(expr.id() == "nearbyint")
  {
    assert(expr.op0().id_string() != "");
    migrate_type(expr.type(), type);

    expr2tc old_expr;
    migrate_expr(expr.op0(), old_expr);

    // Default to rounding mode symbol
    expr2tc rounding_mode =
      expr2tc(new symbol2t(type_pool.get_int32(), "c:@__ESBMC_rounding_mode"));

    // If it's not nil, convert it
    exprt old_rm = expr.find_expr("rounding_mode");
    if(old_rm.is_not_nil())
      migrate_expr(old_rm, rounding_mode);

    nearbyint2t *t = new nearbyint2t(type, old_expr, rounding_mode);
    new_expr_ref = expr2tc(t);
  }
  else if(expr.id() == typet::t_struct)
  {
    migrate_type(expr.type(), type);

    std::vector<expr2tc> members;
    forall_operands(it, expr)
    {
      expr2tc new_ref;
      migrate_expr(*it, new_ref);

      members.push_back(new_ref);
    }

    constant_struct2t *s = new constant_struct2t(type, members);
    new_expr_ref = expr2tc(s);
  }
  else if(expr.id() == typet::t_union)
  {
    // Unions are now being transformed into byte arrays at all stages past
    // parsing.
    new_expr_ref = flatten_union(expr);
  }
  else if(expr.id() == "string-constant")
  {
    std::string thestring = expr.value().as_string();
    typet thetype = expr.type();
    assert(thetype.add(typet::a_size).id() == irept::id_constant);
    exprt &face = (exprt &)thetype.add(typet::a_size);
    BigInt val = binary2bigint(face.value(), false);

    type2tc t = type2tc(new string_type2t(val.to_int64()));

    new_expr_ref = expr2tc(new constant_string2t(t, irep_idt(thestring)));
  }
  else if(
    (expr.id() == irept::id_constant && expr.type().id() == typet::t_array) ||
    expr.id() == typet::t_array)
  {
    // Fixed size array.
    migrate_type(expr.type(), type);

    std::vector<expr2tc> members;
    forall_operands(it, expr)
    {
      expr2tc new_ref;
      migrate_expr(*it, new_ref);

      members.push_back(new_ref);
    }

    constant_array2t *a = new constant_array2t(type, members);
    new_expr_ref = expr2tc(a);
  }
  else if(expr.id() == exprt::arrayof)
  {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 1);
    expr2tc new_value;
    migrate_expr(expr.op0(), new_value);

    constant_array_of2t *a = new constant_array_of2t(type, new_value);
    new_expr_ref = expr2tc(a);
  }
  else if(expr.id() == exprt::i_if)
  {
    migrate_type(expr.type(), type);

    expr2tc cond, true_val, false_val;
    migrate_expr(expr.op0(), cond);
    migrate_expr(expr.op1(), true_val);
    migrate_expr(expr.op2(), false_val);

    if2t *i = new if2t(type, cond, true_val, false_val);
    new_expr_ref = expr2tc(i);
  }
  else if(expr.id() == exprt::equality)
  {
    expr2tc side1, side2;

    convert_operand_pair(expr, side1, side2);

    equality2t *e = new equality2t(side1, side2);
    new_expr_ref = expr2tc(e);
  }
  else if(expr.id() == exprt::notequal)
  {
    expr2tc side1, side2;

    convert_operand_pair(expr, side1, side2);

    notequal2t *n = new notequal2t(side1, side2);
    new_expr_ref = expr2tc(n);
  }
  else if(expr.id() == exprt::i_lt)
  {
    expr2tc side1, side2;

    convert_operand_pair(expr, side1, side2);

    lessthan2t *n = new lessthan2t(side1, side2);
    new_expr_ref = expr2tc(n);
  }
  else if(expr.id() == exprt::i_gt)
  {
    expr2tc side1, side2;
    migrate_expr(expr.op0(), side1);
    migrate_expr(expr.op1(), side2);

    greaterthan2t *n = new greaterthan2t(side1, side2);
    new_expr_ref = expr2tc(n);
  }
  else if(expr.id() == exprt::i_le)
  {
    expr2tc side1, side2;

    convert_operand_pair(expr, side1, side2);

    lessthanequal2t *n = new lessthanequal2t(side1, side2);
    new_expr_ref = expr2tc(n);
  }
  else if(expr.id() == exprt::i_ge)
  {
    expr2tc side1, side2;

    convert_operand_pair(expr, side1, side2);

    greaterthanequal2t *n = new greaterthanequal2t(side1, side2);
    new_expr_ref = expr2tc(n);
  }
  else if(expr.id() == exprt::i_not)
  {
    assert(expr.type().id() == typet::t_bool);
    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    not2t *n = new not2t(theval);
    new_expr_ref = expr2tc(n);
  }
  else if(expr.id() == exprt::i_and)
  {
    assert(expr.type().id() == typet::t_bool);
    expr2tc side1, side2;
    if(expr.operands().size() > 2)
    {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    and2t *a = new and2t(side1, side2);
    new_expr_ref = expr2tc(a);
  }
  else if(expr.id() == exprt::i_or)
  {
    assert(expr.type().id() == typet::t_bool);
    expr2tc side1, side2;

    if(expr.operands().size() > 2)
    {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    or2t *o = new or2t(side1, side2);
    new_expr_ref = expr2tc(o);
  }
  else if(expr.id() == exprt::i_xor)
  {
    assert(expr.type().id() == typet::t_bool);
    assert(expr.operands().size() == 2);
    expr2tc side1, side2;

    convert_operand_pair(expr, side1, side2);

    xor2t *x = new xor2t(side1, side2);
    new_expr_ref = expr2tc(x);
  }
  else if(expr.id() == exprt::implies)
  {
    assert(expr.type().id() == typet::t_bool);
    assert(expr.operands().size() == 2);
    expr2tc side1, side2;

    convert_operand_pair(expr, side1, side2);

    implies2t *i = new implies2t(side1, side2);
    new_expr_ref = expr2tc(i);
  }
  else if(expr.id() == exprt::i_bitand)
  {
    migrate_type(expr.type(), type);

    expr2tc side1, side2;
    if(expr.operands().size() > 2)
    {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    bitand2t *a = new bitand2t(type, side1, side2);
    new_expr_ref = expr2tc(a);
  }
  else if(expr.id() == exprt::i_bitor)
  {
    migrate_type(expr.type(), type);

    expr2tc side1, side2;
    if(expr.operands().size() > 2)
    {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    bitor2t *o = new bitor2t(type, side1, side2);
    new_expr_ref = expr2tc(o);
  }
  else if(expr.id() == exprt::i_bitxor)
  {
    migrate_type(expr.type(), type);

    expr2tc side1, side2;
    if(expr.operands().size() > 2)
    {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    bitxor2t *x = new bitxor2t(type, side1, side2);
    new_expr_ref = expr2tc(x);
  }
  else if(expr.id() == exprt::i_bitnand)
  {
    migrate_type(expr.type(), type);

    expr2tc side1, side2;
    if(expr.operands().size() > 2)
    {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    bitnand2t *n = new bitnand2t(type, side1, side2);
    new_expr_ref = expr2tc(n);
  }
  else if(expr.id() == exprt::i_bitnor)
  {
    migrate_type(expr.type(), type);

    expr2tc side1, side2;
    if(expr.operands().size() > 2)
    {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    bitnor2t *o = new bitnor2t(type, side1, side2);
    new_expr_ref = expr2tc(o);
  }
  else if(expr.id() == exprt::i_bitnxor)
  {
    migrate_type(expr.type(), type);

    expr2tc side1, side2;
    if(expr.operands().size() > 2)
    {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    bitnxor2t *x = new bitnxor2t(type, side1, side2);
    new_expr_ref = expr2tc(x);
  }
  else if(expr.id() == exprt::i_bitnot)
  {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 1);
    expr2tc value;
    migrate_expr(expr.op0(), value);

    bitnot2t *n = new bitnot2t(type, value);
    new_expr_ref = expr2tc(n);
  }
  else if(expr.id() == exprt::i_lshr)
  {
    migrate_type(expr.type(), type);

    expr2tc side1, side2;
    if(expr.operands().size() > 2)
    {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    lshr2t *s = new lshr2t(type, side1, side2);
    new_expr_ref = expr2tc(s);
  }
  else if(expr.id() == "unary-")
  {
    migrate_type(expr.type(), type);

    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    neg2t *n = new neg2t(type, theval);
    new_expr_ref = expr2tc(n);
  }
  else if(expr.id() == exprt::abs)
  {
    migrate_type(expr.type(), type);

    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    abs2t *a = new abs2t(type, theval);
    new_expr_ref = expr2tc(a);
  }
  else if(expr.id() == exprt::plus)
  {
    migrate_type(expr.type(), type);

    expr2tc side1, side2;
    if(expr.operands().size() > 2)
    {
      splice_expr(expr, new_expr_ref);
      return;
    }

    convert_operand_pair(expr, side1, side2);

    add2t *a = new add2t(type, side1, side2);
    new_expr_ref = expr2tc(a);
  }
  else if(expr.id() == exprt::minus)
  {
    migrate_type(expr.type(), type);

    if(expr.operands().size() > 2)
    {
      splice_expr(expr, new_expr_ref);
      return;
    }

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    sub2t *s = new sub2t(type, side1, side2);
    new_expr_ref = expr2tc(s);
  }
  else if(expr.id() == exprt::mult)
  {
    migrate_type(expr.type(), type);

    if(expr.operands().size() > 2)
    {
      splice_expr(expr, new_expr_ref);
      return;
    }

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    mul2t *s = new mul2t(type, side1, side2);
    new_expr_ref = expr2tc(s);
  }
  else if(expr.id() == exprt::div)
  {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 2);

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    div2t *d = new div2t(type, side1, side2);
    new_expr_ref = expr2tc(d);
  }
  else if(expr.id() == "ieee_add")
  {
    migrate_type(expr.type(), type);

    if(expr.operands().size() > 2)
    {
      splice_expr(expr, new_expr_ref);
      return;
    }

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    // Default to rounding mode symbol
    expr2tc rm =
      expr2tc(new symbol2t(type_pool.get_int32(), "c:@__ESBMC_rounding_mode"));

    // If it's not nil, convert it
    exprt old_rm = expr.find_expr("rounding_mode");
    if(old_rm.is_not_nil())
      migrate_expr(old_rm, rm);

    ieee_add2t *a = new ieee_add2t(type, side1, side2, rm);
    new_expr_ref = expr2tc(a);
  }
  else if(expr.id() == "ieee_sub")
  {
    migrate_type(expr.type(), type);

    if(expr.operands().size() > 2)
    {
      splice_expr(expr, new_expr_ref);
      return;
    }

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    // Default to rounding mode symbol
    expr2tc rm =
      expr2tc(new symbol2t(type_pool.get_int32(), "c:@__ESBMC_rounding_mode"));

    // If it's not nil, convert it
    exprt old_rm = expr.find_expr("rounding_mode");
    if(old_rm.is_not_nil())
      migrate_expr(old_rm, rm);

    ieee_sub2t *s = new ieee_sub2t(type, side1, side2, rm);
    new_expr_ref = expr2tc(s);
  }
  else if(expr.id() == "ieee_mul")
  {
    migrate_type(expr.type(), type);

    if(expr.operands().size() > 2)
    {
      splice_expr(expr, new_expr_ref);
      return;
    }

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    // Default to rounding mode symbol
    expr2tc rm =
      expr2tc(new symbol2t(type_pool.get_int32(), "c:@__ESBMC_rounding_mode"));

    // If it's not nil, convert it
    exprt old_rm = expr.find_expr("rounding_mode");
    if(old_rm.is_not_nil())
      migrate_expr(old_rm, rm);

    ieee_mul2t *s = new ieee_mul2t(type, side1, side2, rm);
    new_expr_ref = expr2tc(s);
  }
  else if(expr.id() == "ieee_div")
  {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 2);

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    // Default to rounding mode symbol
    expr2tc rm =
      expr2tc(new symbol2t(type_pool.get_int32(), "c:@__ESBMC_rounding_mode"));

    // If it's not nil, convert it
    exprt old_rm = expr.find_expr("rounding_mode");
    if(old_rm.is_not_nil())
      migrate_expr(old_rm, rm);

    ieee_div2t *d = new ieee_div2t(type, side1, side2, rm);
    new_expr_ref = expr2tc(d);
  }
  else if(expr.id() == "ieee_fma")
  {
    migrate_type(expr.type(), type);

    expr2tc v1, v2, v3;
    migrate_expr(expr.op0(), v1);
    migrate_expr(expr.op1(), v2);
    migrate_expr(expr.op2(), v3);

    // Default to rounding mode symbol
    expr2tc rm =
      expr2tc(new symbol2t(type_pool.get_int32(), "c:@__ESBMC_rounding_mode"));

    // If it's not nil, convert it
    exprt old_rm = expr.find_expr("rounding_mode");
    if(old_rm.is_not_nil())
      migrate_expr(old_rm, rm);

    ieee_fma2t *a = new ieee_fma2t(type, v1, v2, v3, rm);
    new_expr_ref = expr2tc(a);
  }
  else if(expr.id() == "ieee_sqrt")
  {
    migrate_type(expr.type(), type);

    expr2tc value;
    migrate_expr(expr.op0(), value);

    // Default to rounding mode symbol
    expr2tc rm =
      expr2tc(new symbol2t(type_pool.get_int32(), "c:@__ESBMC_rounding_mode"));

    // If it's not nil, convert it
    exprt old_rm = expr.find_expr("rounding_mode");
    if(old_rm.is_not_nil())
      migrate_expr(old_rm, rm);

    ieee_sqrt2t *a = new ieee_sqrt2t(type, value, rm);
    new_expr_ref = expr2tc(a);
  }
  else if(expr.id() == exprt::mod)
  {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 2);

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    modulus2t *m = new modulus2t(type, side1, side2);
    new_expr_ref = expr2tc(m);
  }
  else if(expr.id() == exprt::i_shl)
  {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 2);

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    shl2t *s = new shl2t(type, side1, side2);
    new_expr_ref = expr2tc(s);
  }
  else if(expr.id() == exprt::i_ashr)
  {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 2);

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    ashr2t *a = new ashr2t(type, side1, side2);
    new_expr_ref = expr2tc(a);
  }
  else if(expr.id() == "pointer_offset")
  {
    migrate_type(expr.type(), type);

    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    pointer_offset2t *p = new pointer_offset2t(type, theval);
    new_expr_ref = expr2tc(p);
  }
  else if(expr.id() == "pointer_object")
  {
    migrate_type(expr.type(), type);

    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    pointer_object2t *p = new pointer_object2t(type, theval);
    new_expr_ref = expr2tc(p);
  }
  else if(expr.id() == exprt::id_address_of)
  {
    assert(expr.type().id() == typet::t_pointer);

    migrate_type(expr.type().subtype(), type);

    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    address_of2t *a = new address_of2t(type, theval);
    new_expr_ref = expr2tc(a);
  }
  else if(
    expr.id() == "byte_extract_little_endian" ||
    expr.id() == "byte_extract_big_endian")
  {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 2);

    expr2tc side1, side2;
    convert_operand_pair(expr, side1, side2);

    bool big_endian = (expr.id() == "byte_extract_big_endian") ? true : false;

    byte_extract2t *b = new byte_extract2t(type, side1, side2, big_endian);
    new_expr_ref = expr2tc(b);
  }
  else if(
    expr.id() == "byte_update_little_endian" ||
    expr.id() == "byte_update_big_endian")
  {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 3);

    expr2tc sourceval, offs;
    convert_operand_pair(expr, sourceval, offs);

    expr2tc update;
    migrate_expr(expr.op2(), update);

    bool big_endian = (expr.id() == "byte_update_big_endian") ? true : false;

    byte_update2t *u =
      new byte_update2t(type, sourceval, offs, update, big_endian);
    new_expr_ref = expr2tc(u);
  }
  else if(expr.id() == "with")
  {
    migrate_type(expr.type(), type);

    expr2tc sourcedata, idx;
    migrate_expr(expr.op0(), sourcedata);

    if(expr.op1().id() == "member_name")
    {
      idx = expr2tc(new constant_string2t(
        type2tc(new string_type2t(1)),
        expr.op1().get_string("component_name")));
    }
    else
    {
      migrate_expr(expr.op1(), idx);
    }

    expr2tc update;
    migrate_expr(expr.op2(), update);

    with2t *w = new with2t(type, sourcedata, idx, update);
    new_expr_ref = expr2tc(w);
  }
  else if(expr.id() == exprt::member)
  {
    migrate_type(expr.type(), type);

    expr2tc sourcedata;
    migrate_expr(expr.op0(), sourcedata);

    member2t *m = new member2t(type, sourcedata, expr.component_name());
    new_expr_ref = expr2tc(m);
  }
  else if(expr.id() == exprt::index)
  {
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 2);
    expr2tc source, index;
    convert_operand_pair(expr, source, index);

    index2t *i = new index2t(type, source, index);
    new_expr_ref = expr2tc(i);
  }
  else if(expr.id() == "memory-leak")
  {
    // Memory leaks are in fact selects/indexes.
    migrate_type(expr.type(), type);

    assert(expr.operands().size() == 2);
    assert(expr.type().id() == typet::t_bool);
    expr2tc source, index;
    convert_operand_pair(expr, source, index);

    index2t *i = new index2t(type, source, index);
    new_expr_ref = expr2tc(i);
  }
  else if(expr.id() == exprt::isnan)
  {
    assert(expr.operands().size() == 1);

    expr2tc val;
    migrate_expr(expr.op0(), val);

    isnan2t *i = new isnan2t(val);
    new_expr_ref = expr2tc(i);
  }
  else if(expr.id() == irept::a_width)
  {
    assert(expr.operands().size() == 1);
    migrate_type(expr.type(), type);

    uint64_t thewidth = type->get_width();
    type2tc inttype(new unsignedbv_type2t(config.ansi_c.int_width));
    new_expr_ref = expr2tc(new constant_int2t(inttype, BigInt(thewidth)));
  }
  else if(expr.id() == "same-object")
  {
    assert(expr.operands().size() == 2);
    assert(expr.type().id() == typet::t_bool);
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);

    same_object2t *s = new same_object2t(op0, op1);
    new_expr_ref = expr2tc(s);
  }
  else if(expr.id() == "invalid-object")
  {
    assert(expr.type().id() == "pointer");
    type2tc pointertype(new pointer_type2t(type2tc(new empty_type2t())));
    new_expr_ref = expr2tc(new symbol2t(pointertype, "INVALID"));
  }
  else if(expr.id() == "unary+")
  {
    migrate_expr(expr.op0(), new_expr_ref);
  }
  else if(expr.id() == "overflow-+")
  {
    assert(expr.type().id() == typet::t_bool);
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);
    expr2tc add = expr2tc(new add2t(op0->type, op0, op1)); // XXX type?
    new_expr_ref = expr2tc(new overflow2t(add));
  }
  else if(expr.id() == "overflow--")
  {
    assert(expr.type().id() == typet::t_bool);
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);
    expr2tc sub = expr2tc(new sub2t(op0->type, op0, op1)); // XXX type?
    new_expr_ref = expr2tc(new overflow2t(sub));
  }
  else if(expr.id() == "overflow-*")
  {
    assert(expr.type().id() == typet::t_bool);
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);
    expr2tc mul = expr2tc(new mul2t(op0->type, op0, op1)); // XXX type?
    new_expr_ref = expr2tc(new overflow2t(mul));
  }
  else if(expr.id() == "overflow-/")
  {
    assert(expr.type().id() == typet::t_bool);
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);
    expr2tc div = expr2tc(new div2t(op0->type, op0, op1)); // XXX type?
    new_expr_ref = expr2tc(new overflow2t(div));
  }
  else if(expr.id() == "overflow-mod")
  {
    assert(expr.type().id() == typet::t_bool);
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);
    expr2tc mod = expr2tc(new modulus2t(op0->type, op0, op1)); // XXX type?
    new_expr_ref = expr2tc(new overflow2t(mod));
  }
  else if(expr.id() == "overflow-shl")
  {
    assert(expr.type().id() == typet::t_bool);
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);
    expr2tc shl = expr2tc(new shl2t(op0->type, op0, op1)); // XXX type?
    new_expr_ref = expr2tc(new overflow2t(shl));
  }
  else if(has_prefix(expr.id_string(), "overflow-typecast-"))
  {
    unsigned bits = atoi(expr.id_string().c_str() + 18);
    expr2tc operand;
    migrate_expr(expr.op0(), operand);
    new_expr_ref = expr2tc(new overflow_cast2t(operand, bits));
  }
  else if(expr.id() == "overflow-unary-")
  {
    assert(expr.type().id() == typet::t_bool);
    expr2tc operand;
    migrate_expr(expr.op0(), operand);
    new_expr_ref = expr2tc(new overflow_neg2t(operand));
  }
  else if(expr.id() == "unknown")
  {
    migrate_type(expr.type(), type);
    new_expr_ref = expr2tc(new unknown2t(type));
  }
  else if(expr.id() == "invalid")
  {
    migrate_type(expr.type(), type);
    new_expr_ref = expr2tc(new invalid2t(type));
  }
  else if(expr.id() == "NULL-object")
  {
    migrate_type(expr.type(), type);
    new_expr_ref = expr2tc(new null_object2t(type));
  }
  else if(expr.id() == "dynamic_object")
  {
    migrate_type(expr.type(), type);
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);

    bool invalid = false;
    bool unknown = false;
    if(is_constant_bool2t(op1))
    {
      invalid = to_constant_bool2t(op1).value;
    }
    else
    {
      assert(expr.op1().id() == "unknown");
      unknown = true;
    }

    new_expr_ref = expr2tc(new dynamic_object2t(type, op0, invalid, unknown));
  }
  else if(expr.id() == irept::id_dereference)
  {
    migrate_type(expr.type(), type);
    expr2tc op0;
    migrate_expr(expr.op0(), op0);
    new_expr_ref = expr2tc(new dereference2t(type, op0));
  }
  else if(expr.id() == "valid_object")
  {
    expr2tc op0;
    migrate_expr(expr.op0(), op0);
    new_expr_ref = expr2tc(new valid_object2t(op0));
  }
  else if(expr.id() == "deallocated_object")
  {
    expr2tc op0;
    migrate_expr(expr.op0(), op0);
    new_expr_ref = expr2tc(new deallocated_obj2t(op0));
  }
  else if(expr.id() == "dynamic_size")
  {
    expr2tc op0;
    migrate_expr(expr.op0(), op0);
    new_expr_ref = expr2tc(new dynamic_size2t(op0));
  }
  else if(expr.id() == "sideeffect")
  {
    expr2tc operand, thesize;
    type2tc cmt_type, plaintype;
    std::vector<expr2tc> args;
    if(
      expr.statement() != "nondet" && expr.statement() != "cpp_new" &&
      expr.statement() != "cpp_new[]")
      migrate_expr(expr.op0(), operand);

    if(expr.statement() == "cpp_new" || expr.statement() == "cpp_new[]")
      // These hide the size in a real size field,
      migrate_expr((const exprt &)expr.find("size"), thesize);
    else if(expr.statement() != "nondet" && expr.statement() != "function_call")
      // For everything other than nondet,
      migrate_expr((const exprt &)expr.cmt_size(), thesize);

    migrate_type((const typet &)expr.cmt_type(), cmt_type);
    migrate_type(expr.type(), plaintype);

    sideeffect2t::allockind t;
    if(expr.statement() == "malloc")
      t = sideeffect2t::malloc;
    else if(expr.statement() == "realloc")
      t = sideeffect2t::realloc;
    else if(expr.statement() == "alloca")
      t = sideeffect2t::alloca;
    else if(expr.statement() == "cpp_new")
      t = sideeffect2t::cpp_new;
    else if(expr.statement() == "cpp_new[]")
      t = sideeffect2t::cpp_new_arr;
    else if(expr.statement() == "nondet")
      t = sideeffect2t::nondet;
    else if(expr.statement() == "va_arg")
      t = sideeffect2t::va_arg;
    else if(expr.statement() == "function_call")
    {
      t = sideeffect2t::function_call;
      const exprt &arguments = expr.op1();
      forall_operands(it, arguments)
      {
        args.emplace_back();
        migrate_expr(*it, args.back());
      }
    }
    else if(expr.statement() == "printf")
    {
      for(auto const &it : expr.operands())
      {
        expr2tc tmp_op;
        migrate_expr(it, tmp_op);
        args.push_back(tmp_op);
      }
      new_expr_ref = code_printf2tc(args);
      return;
    }
    else if(expr.statement() == "preincrement")
    {
      t = sideeffect2t::preincrement;
      migrate_expr(expr.op0(), new_expr_ref);
    }
    else if(expr.statement() == "postincrement")
    {
      t = sideeffect2t::postincrement;
      migrate_expr(expr.op0(), new_expr_ref);
    }
    else if(expr.statement() == "predecrement")
    {
      t = sideeffect2t::predecrement;
      migrate_expr(expr.op0(), new_expr_ref);
    }
    else if(expr.statement() == "postdecrement")
    {
      t = sideeffect2t::predecrement;
      migrate_expr(expr.op0(), new_expr_ref);
    }
    else
    {
      std::cerr << "Unexpected side-effect statement: " << expr.statement()
                << std::endl;
      abort();
    }

    new_expr_ref =
      sideeffect2tc(plaintype, operand, thesize, args, cmt_type, t);
  }
  else if(expr.id() == irept::id_code && expr.statement() == "assign")
  {
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);
    new_expr_ref = expr2tc(new code_assign2t(op0, op1));
  }
  else if(expr.id() == irept::id_code && expr.statement() == "decl")
  {
    assert(expr.op0().id() == "symbol");
    type2tc thetype;
    irep_idt sym_name;
    migrate_type(expr.op0().type(), thetype);
    sym_name = expr.op0().identifier();
    new_expr_ref = expr2tc(new code_decl2t(thetype, sym_name));
  }
  else if(expr.id() == irept::id_code && expr.statement() == "dead")
  {
    assert(expr.op0().id() == "symbol");
    type2tc thetype;
    irep_idt sym_name;
    migrate_type(expr.op0().type(), thetype);
    sym_name = expr.op0().identifier();
    new_expr_ref = expr2tc(new code_dead2t(thetype, sym_name));
  }
  else if(expr.id() == irept::id_code && expr.statement() == "printf")
  {
    std::vector<expr2tc> ops;
    for(auto const &it : expr.operands())
    {
      expr2tc tmp_op;
      migrate_expr(it, tmp_op);
      ops.push_back(tmp_op);
    }
    new_expr_ref = expr2tc(new code_printf2t(ops));
  }
  else if(expr.id() == irept::id_code && expr.statement() == "expression")
  {
    assert(expr.operands().size() == 1);
    expr2tc theop;
    migrate_expr(expr.op0(), theop);
    new_expr_ref = expr2tc(new code_expression2t(theop));
  }
  else if(expr.id() == irept::id_code && expr.statement() == "return")
  {
    expr2tc theop;
    if(expr.operands().size() == 1)
      migrate_expr(expr.op0(), theop);
    else
      assert(expr.operands().size() == 0);
    new_expr_ref = expr2tc(new code_return2t(theop));
  }
  else if(expr.id() == irept::id_code && expr.statement() == "free")
  {
    assert(expr.operands().size() == 1);
    expr2tc theop;
    migrate_expr(expr.op0(), theop);
    new_expr_ref = expr2tc(new code_free2t(theop));
  }
  else if(expr.id() == irept::id_code && expr.statement() == "cpp_delete[]")
  {
    assert(expr.operands().size() == 1);
    expr2tc theop;
    migrate_expr(expr.op0(), theop);
    new_expr_ref = expr2tc(new code_cpp_del_array2t(theop));
  }
  else if(expr.id() == irept::id_code && expr.statement() == "cpp_delete")
  {
    assert(expr.operands().size() == 1);
    expr2tc theop;
    migrate_expr(expr.op0(), theop);
    new_expr_ref = expr2tc(new code_cpp_delete2t(theop));
  }
  else if(expr.id() == "object_descriptor")
  {
    migrate_type(expr.op0().type(), type);
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);
    new_expr_ref = expr2tc(new object_descriptor2t(type, op0, op1, 0));
  }
  else if(expr.id() == irept::id_code && expr.statement() == "function_call")
  {
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);

    std::vector<expr2tc> args;
    const exprt &irep_args = expr.op2();
    assert(irep_args.is_not_nil());
    forall_operands(it, irep_args)
    {
      expr2tc tmp;
      migrate_expr(*it, tmp);
      args.push_back(tmp);
    }

    new_expr_ref = expr2tc(new code_function_call2t(op0, op1, args));
  }
  else if(expr.id() == "invalid-pointer")
  {
    expr2tc op0;
    migrate_expr(expr.op0(), op0);
    new_expr_ref = expr2tc(new invalid_pointer2t(op0));
  }
  else if(expr.id() == "code" && expr.statement() == "skip")
  {
    new_expr_ref = expr2tc(new code_skip2t(get_empty_type()));
  }
  else if(expr.id() == "code" && expr.statement() == "goto")
  {
    new_expr_ref = expr2tc(new code_goto2t(expr.get("destination")));
  }
  else if(expr.id() == "comma")
  {
    migrate_type(expr.type(), type);
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);
    new_expr_ref = expr2tc(new code_comma2t(type, op0, op1));
  }
  else if(expr.id() == "code" && expr.statement() == "asm")
  {
    migrate_type(expr.type(), type);
    const irep_idt &str = expr.op0().value();
    new_expr_ref = expr2tc(new code_asm2t(type, str));
  }
  else if(expr.id() == "code" && expr.statement() == "cpp-throw")
  {
    // No type,
    const irept::subt &exceptions_thrown =
      expr.find("exception_list").get_sub();

    std::vector<irep_idt> expr_list;
    for(const auto &e_it : exceptions_thrown)
    {
      expr_list.push_back(e_it.id());
    }

    expr2tc operand;
    if(expr.operands().size() == 1)
    {
      migrate_expr(expr.op0(), operand);
    }
    else
    {
      operand = expr2tc();
    }

    new_expr_ref = expr2tc(new code_cpp_throw2t(operand, expr_list));
  }
  else if(expr.id() == "code" && expr.statement() == "throw-decl")
  {
    std::vector<irep_idt> expr_list;
    const irept::subt &exceptions_thrown = expr.find("throw_list").get_sub();
    for(const auto &e_it : exceptions_thrown)
    {
      expr_list.push_back(e_it.id());
    }

    new_expr_ref = expr2tc(new code_cpp_throw_decl2t(expr_list));
  }
  else if(expr.id() == "isinf")
  {
    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    isinf2t *n = new isinf2t(theval);
    new_expr_ref = expr2tc(n);
  }
  else if(expr.id() == "isnormal")
  {
    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    isnormal2t *n = new isnormal2t(theval);
    new_expr_ref = expr2tc(n);
  }
  else if(expr.id() == "isfinite")
  {
    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    isfinite2t *n = new isfinite2t(theval);
    new_expr_ref = expr2tc(n);
  }
  else if(expr.id() == "signbit")
  {
    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    signbit2t *n = new signbit2t(theval);
    new_expr_ref = expr2tc(n);
  }
  else if(expr.id() == "popcount")
  {
    expr2tc theval;
    migrate_expr(expr.op0(), theval);

    popcount2t *n = new popcount2t(theval);
    new_expr_ref = expr2tc(n);
  }
  else if(expr.id() == "bswap")
  {
    expr2tc theval;
    migrate_expr(expr.op0(), theval);
    migrate_type(expr.type(), type);

    bswap2t *n = new bswap2t(type, theval);
    new_expr_ref = expr2tc(n);
  }
  else if(expr.id() == "concat")
  {
    expr2tc op0, op1;
    convert_operand_pair(expr, op0, op1);
    migrate_type(expr.type(), type);
    new_expr_ref = concat2tc(type, op0, op1);
  }
  else if(expr.id() == "extract")
  {
    expr2tc theop;
    migrate_type(expr.type(), type);
    migrate_expr(expr.op0(), theop);
    unsigned int upper = atoi(expr.get("upper").as_string().c_str());
    unsigned int lower = atoi(expr.get("lower").as_string().c_str());
    new_expr_ref = extract2tc(type, theop, upper, lower);
  }
  else
  {
    expr.dump();
    throw new std::string("migrate expr failed");
  }
}

typet migrate_type_back(const type2tc &ref)
{
  switch(ref->type_id)
  {
  case type2t::bool_id:
    return bool_typet();
  case type2t::empty_id:
    return empty_typet();
  case type2t::symbol_id:
  {
    const symbol_type2t &ref2 = to_symbol_type(ref);
    return symbol_typet(ref2.symbol_name);
  }
  case type2t::struct_id:
  {
    unsigned int idx;
    struct_typet thetype;
    struct_union_typet::componentst comps;
    const struct_type2t &ref2 = to_struct_type(ref);

    idx = 0;
    for(auto const &it : ref2.members)
    {
      struct_union_typet::componentt component;
      component.id("component");
      component.type() = migrate_type_back(it);
      component.set_name(irep_idt(ref2.member_names[idx]));
      component.pretty_name(irep_idt(ref2.member_pretty_names[idx]));
      comps.push_back(component);
      idx++;
    }

    thetype.components() = comps;
    thetype.set("tag", irep_idt(ref2.name));
    if(ref2.packed)
      thetype.set("packed", irep_idt("true"));
    return thetype;
  }
  case type2t::union_id:
  {
    unsigned int idx;
    union_typet thetype;
    struct_union_typet::componentst comps;
    const union_type2t &ref2 = to_union_type(ref);

    idx = 0;
    for(auto const &it : ref2.members)
    {
      struct_union_typet::componentt component;
      component.id("component");
      component.type() = migrate_type_back(it);
      component.set_name(irep_idt(ref2.member_names[idx]));
      component.pretty_name(irep_idt(ref2.member_pretty_names[idx]));
      comps.push_back(component);
      idx++;
    }

    thetype.components() = comps;
    thetype.set("tag", irep_idt(ref2.name));
    return thetype;
  }
  case type2t::code_id:
  {
    const code_type2t &ref2 = to_code_type(ref);
    code_typet code;
    typet ret_type = migrate_type_back(ref2.ret_type);

    assert(ref2.arguments.size() == ref2.argument_names.size());

    code_typet::argumentst args;
    unsigned int i = 0;
    for(auto const &it : ref2.arguments)
    {
      args.emplace_back(migrate_type_back(it));
      args.back().set_identifier(ref2.argument_names[i]);
      i++;
    }

    code.arguments() = args;
    code.return_type() = ret_type;

    if(ref2.ellipsis)
      code.make_ellipsis();

    return code;
  }
  case type2t::array_id:
  {
    const array_type2t &ref2 = to_array_type(ref);

    array_typet thetype;
    thetype.subtype() = migrate_type_back(ref2.subtype);
    if(ref2.size_is_infinite)
    {
      thetype.set("size", "infinity");
    }
    else
    {
      thetype.size() = migrate_expr_back(ref2.array_size);
    }

    return thetype;
  }
  case type2t::pointer_id:
  {
    const pointer_type2t &ref2 = to_pointer_type(ref);

    typet subtype = migrate_type_back(ref2.subtype);
    pointer_typet thetype(subtype);
    return thetype;
  }
  case type2t::unsignedbv_id:
  {
    const unsignedbv_type2t &ref2 = to_unsignedbv_type(ref);

    return unsignedbv_typet(ref2.width);
  }
  case type2t::signedbv_id:
  {
    const signedbv_type2t &ref2 = to_signedbv_type(ref);

    return signedbv_typet(ref2.width);
  }
  case type2t::fixedbv_id:
  {
    const fixedbv_type2t &ref2 = to_fixedbv_type(ref);

    fixedbv_typet thetype;
    thetype.set_integer_bits(ref2.integer_bits);
    thetype.set_width(ref2.width);
    return thetype;
  }
  case type2t::floatbv_id:
  {
    const floatbv_type2t &ref2 = to_floatbv_type(ref);

    floatbv_typet thetype;
    thetype.set_f(ref2.fraction);
    thetype.set_width(ref2.get_width());
    return thetype;
  }
  case type2t::string_id:
  {
    string_typet ret;
    ret.width(to_string_type(ref).get_length());
    return ret;
  }
  case type2t::cpp_name_id:
  {
    const cpp_name_type2t &ref2 = to_cpp_name_type(ref);
    exprt thetype("cpp-name");
    exprt name("name");
    name.identifier(ref2.name);
    thetype.get_sub().push_back(name);

    if(ref2.template_args.size() != 0)
    {
      exprt args("template_args");
      exprt &arglist = (exprt &)args.add("arguments");
      for(auto const &it : ref2.template_args)
      {
        typet tmp = migrate_type_back(it);
        exprt type("type");
        type.type() = tmp;
        arglist.copy_to_operands(type); // Yep, that's how it's structured.
      }

      thetype.get_sub().push_back(args);
    }

    typet ret;
    ret.swap((irept &)thetype);
    return ret;
  }
  default:
    std::cerr << "Unrecognized type in migrate_type_back" << std::endl;
    abort();
  }
}

exprt migrate_expr_back(const expr2tc &ref)
{
  if(ref.get() == nullptr)
    return nil_exprt();

  switch(ref->expr_id)
  {
  case expr2t::constant_int_id:
  {
    const constant_int2t &ref2 = to_constant_int2t(ref);
    typet thetype = migrate_type_back(ref->type);
    constant_exprt theexpr(thetype);
    unsigned int width = atoi(thetype.width().as_string().c_str());
    theexpr.set_value(integer2binary(ref2.value, width));
    return theexpr;
  }
  case expr2t::constant_fixedbv_id:
  {
    return to_constant_fixedbv2t(ref).value.to_expr();
  }
  case expr2t::constant_floatbv_id:
  {
    return to_constant_floatbv2t(ref).value.to_expr();
  }
  case expr2t::constant_bool_id:
  {
    const constant_bool2t &ref2 = to_constant_bool2t(ref);
    if(ref2.value)
      return true_exprt();

    return false_exprt();
  }
  case expr2t::constant_string_id:
  {
    const constant_string2t &ref2 = to_constant_string2t(ref);
    const string_type2t &typeref = to_string_type(ref->type);
    exprt thestring("string-constant");

    typet thetype("array");
    thetype.subtype() = signedbv_typet(8);
    constant_exprt sizeexpr(signedbv_typet(32));
    sizeexpr.set("value", integer2binary(BigInt(typeref.width), 32));
    thetype.size(sizeexpr);

    thestring.type() = thetype;
    thestring.set("value", irep_idt(ref2.value));
    return thestring;
  }
  case expr2t::constant_struct_id:
  {
    const constant_struct2t &ref2 = to_constant_struct2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt thestruct("struct", thetype);
    for(auto const &it : ref2.datatype_members)
      thestruct.operands().push_back(migrate_expr_back(it));
    return thestruct;
  }
  case expr2t::constant_union_id:
  {
    const constant_union2t &ref2 = to_constant_union2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt theunion("union", thetype);
    for(auto const &it : ref2.datatype_members)
      theunion.operands().push_back(migrate_expr_back(it));
    return theunion;
  }
  case expr2t::constant_array_id:
  {
    const constant_array2t &ref2 = to_constant_array2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt thearray("constant", thetype);
    for(auto const &it : ref2.datatype_members)
      thearray.operands().push_back(migrate_expr_back(it));
    return thearray;
  }
  case expr2t::constant_array_of_id:
  {
    const constant_array_of2t &ref2 = to_constant_array_of2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt thearray("array_of", thetype);
    exprt initializer = migrate_expr_back(ref2.initializer);
    thearray.operands().push_back(initializer);
    return thearray;
  }
  case expr2t::symbol_id:
  {
    const symbol2t &ref2 = to_symbol2t(ref);
    typet thetype = migrate_type_back(ref->type);
    if(has_prefix(ref2.thename.as_string(), "nondet$"))
    {
      exprt thesym("nondet_symbol", thetype);
      thesym.identifier(
        irep_idt(std::string(ref2.get_symbol_name().c_str() + 7)));
      return thesym;
    }
    if(ref2.thename == "NULL")
    {
      // Special case.
      constant_exprt const_expr(migrate_type_back(ref2.type));
      const_expr.set_value(ref2.get_symbol_name());
      return const_expr;
    }
    else if(ref2.thename == "INVALID")
    {
      exprt invalid("invalid-object", pointer_typet(empty_typet()));
      return invalid;
    }
    else
    {
      return symbol_exprt(ref2.get_symbol_name(), thetype);
    }
  }
  case expr2t::typecast_id:
  {
    const typecast2t &ref2 = to_typecast2t(ref);
    typet thetype = migrate_type_back(ref->type);

    typecast_exprt new_expr(migrate_expr_back(ref2.from), thetype);
    new_expr.set("rounding_mode", migrate_expr_back(ref2.rounding_mode));
    return new_expr;
  }
  case expr2t::nearbyint_id:
  {
    const nearbyint2t &ref2 = to_nearbyint2t(ref);
    typet thetype = migrate_type_back(ref->type);

    exprt new_expr("nearbyint", thetype);
    new_expr.copy_to_operands(migrate_expr_back(ref2.from));
    new_expr.set("rounding_mode", migrate_expr_back(ref2.rounding_mode));
    return new_expr;
  }
  case expr2t::if_id:
  {
    const if2t &ref2 = to_if2t(ref);
    typet thetype = migrate_type_back(ref->type);
    if_exprt theif(
      migrate_expr_back(ref2.cond),
      migrate_expr_back(ref2.true_value),
      migrate_expr_back(ref2.false_value));
    theif.type() = thetype;
    return theif;
  }
  case expr2t::equality_id:
  {
    const equality2t &ref2 = to_equality2t(ref);
    return equality_exprt(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
  }
  case expr2t::notequal_id:
  {
    const notequal2t &ref2 = to_notequal2t(ref);
    exprt notequal("notequal", bool_typet());
    notequal.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return notequal;
  }
  case expr2t::lessthan_id:
  {
    const lessthan2t &ref2 = to_lessthan2t(ref);
    exprt lessthan("<", bool_typet());
    lessthan.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return lessthan;
  }
  case expr2t::greaterthan_id:
  {
    const greaterthan2t &ref2 = to_greaterthan2t(ref);
    exprt greaterthan(">", bool_typet());
    greaterthan.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return greaterthan;
  }
  case expr2t::lessthanequal_id:
  {
    const lessthanequal2t &ref2 = to_lessthanequal2t(ref);
    exprt lessthanequal("<=", bool_typet());
    lessthanequal.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return lessthanequal;
  }
  case expr2t::greaterthanequal_id:
  {
    const greaterthanequal2t &ref2 = to_greaterthanequal2t(ref);
    exprt greaterthanequal(">=", bool_typet());
    greaterthanequal.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return greaterthanequal;
  }
  case expr2t::not_id:
  {
    const not2t &ref2 = to_not2t(ref);
    return not_exprt(migrate_expr_back(ref2.value));
  }
  case expr2t::and_id:
  {
    const and2t &ref2 = to_and2t(ref);
    exprt andval("and", bool_typet());
    andval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return andval;
  }
  case expr2t::or_id:
  {
    const or2t &ref2 = to_or2t(ref);
    exprt orval("or", bool_typet());
    orval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return orval;
  }
  case expr2t::xor_id:
  {
    const xor2t &ref2 = to_xor2t(ref);
    exprt xorval("xor", bool_typet());
    xorval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return xorval;
  }
  case expr2t::implies_id:
  {
    const implies2t &ref2 = to_implies2t(ref);
    exprt impliesval("=>", bool_typet());
    impliesval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return impliesval;
  }
  case expr2t::bitand_id:
  {
    const bitand2t &ref2 = to_bitand2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt bitandval("bitand", thetype);
    bitandval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return bitandval;
  }
  case expr2t::bitor_id:
  {
    const bitor2t &ref2 = to_bitor2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt bitorval("bitor", thetype);
    bitorval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return bitorval;
  }
  case expr2t::bitxor_id:
  {
    const bitxor2t &ref2 = to_bitxor2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt bitxorval("bitxor", thetype);
    bitxorval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return bitxorval;
  }
  case expr2t::bitnand_id:
  {
    const bitnand2t &ref2 = to_bitnand2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt bitnandval("bitnand", thetype);
    bitnandval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return bitnandval;
  }
  case expr2t::bitnor_id:
  {
    const bitnor2t &ref2 = to_bitnor2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt bitnorval("bitnor", thetype);
    bitnorval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return bitnorval;
  }
  case expr2t::bitnxor_id:
  {
    const bitnxor2t &ref2 = to_bitnxor2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt bitnxorval("bitnxor", thetype);
    bitnxorval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return bitnxorval;
  }
  case expr2t::bitnot_id:
  {
    const bitnot2t &ref2 = to_bitnot2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt bitnotval("bitnot", thetype);
    bitnotval.copy_to_operands(migrate_expr_back(ref2.value));
    return bitnotval;
  }
  case expr2t::lshr_id:
  {
    const lshr2t &ref2 = to_lshr2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt lshrval("lshr", thetype);
    lshrval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return lshrval;
  }
  case expr2t::neg_id:
  {
    const neg2t &ref2 = to_neg2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt neg("unary-", thetype);
    neg.copy_to_operands(migrate_expr_back(ref2.value));
    return neg;
  }
  case expr2t::abs_id:
  {
    const abs2t &ref2 = to_abs2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt abs("abs", thetype);
    abs.copy_to_operands(migrate_expr_back(ref2.value));
    return abs;
  }
  case expr2t::add_id:
  {
    const add2t &ref2 = to_add2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt addval("+", thetype);
    addval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return addval;
  }
  case expr2t::sub_id:
  {
    const sub2t &ref2 = to_sub2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt subval("-", thetype);
    subval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return subval;
  }
  case expr2t::mul_id:
  {
    const mul2t &ref2 = to_mul2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt mulval("*", thetype);
    mulval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return mulval;
  }
  case expr2t::div_id:
  {
    const div2t &ref2 = to_div2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt divval("/", thetype);
    divval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return divval;
  }
  case expr2t::ieee_add_id:
  {
    const ieee_add2t &ref2 = to_ieee_add2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt addval("ieee_add", thetype);
    addval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));

    // Add rounding mode
    addval.set("rounding_mode", migrate_expr_back(ref2.rounding_mode));
    return addval;
  }
  case expr2t::ieee_sub_id:
  {
    const ieee_sub2t &ref2 = to_ieee_sub2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt subval("ieee_sub", thetype);
    subval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));

    // Add rounding mode
    subval.set("rounding_mode", migrate_expr_back(ref2.rounding_mode));
    return subval;
  }
  case expr2t::ieee_mul_id:
  {
    const ieee_mul2t &ref2 = to_ieee_mul2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt mulval("ieee_mul", thetype);
    mulval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));

    // Add rounding mode
    mulval.set("rounding_mode", migrate_expr_back(ref2.rounding_mode));
    return mulval;
  }
  case expr2t::ieee_div_id:
  {
    const ieee_div2t &ref2 = to_ieee_div2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt divval("ieee_div", thetype);
    divval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));

    // Add rounding mode
    divval.set("rounding_mode", migrate_expr_back(ref2.rounding_mode));
    return divval;
  }
  case expr2t::ieee_fma_id:
  {
    const ieee_fma2t &ref2 = to_ieee_fma2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt fmaval("ieee_fma", thetype);
    fmaval.copy_to_operands(
      migrate_expr_back(ref2.value_1),
      migrate_expr_back(ref2.value_2),
      migrate_expr_back(ref2.value_3));

    // Add rounding mode
    fmaval.set("rounding_mode", migrate_expr_back(ref2.rounding_mode));
    return fmaval;
  }
  case expr2t::ieee_sqrt_id:
  {
    const ieee_sqrt2t &ref2 = to_ieee_sqrt2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt fmaval("ieee_sqrt", thetype);
    fmaval.copy_to_operands(migrate_expr_back(ref2.value));

    // Add rounding mode
    fmaval.set("rounding_mode", migrate_expr_back(ref2.rounding_mode));
    return fmaval;
  }
  case expr2t::modulus_id:
  {
    const modulus2t &ref2 = to_modulus2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt modval("mod", thetype);
    modval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return modval;
  }
  case expr2t::shl_id:
  {
    const shl2t &ref2 = to_shl2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt shlval("shl", thetype);
    shlval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return shlval;
  }
  case expr2t::ashr_id:
  {
    const ashr2t &ref2 = to_ashr2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt ashrval("ashr", thetype);
    ashrval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return ashrval;
  }
  case expr2t::same_object_id:
  {
    const same_object2t &ref2 = to_same_object2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt same_objectval("same-object", thetype);
    same_objectval.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return same_objectval;
  }
  case expr2t::pointer_offset_id:
  {
    const pointer_offset2t &ref2 = to_pointer_offset2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt pointer_offsetval("pointer_offset", thetype);
    pointer_offsetval.copy_to_operands(migrate_expr_back(ref2.ptr_obj));
    return pointer_offsetval;
  }
  case expr2t::pointer_object_id:
  {
    const pointer_object2t &ref2 = to_pointer_object2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt pointer_objectval("pointer_object", thetype);
    pointer_objectval.copy_to_operands(migrate_expr_back(ref2.ptr_obj));
    return pointer_objectval;
  }
  case expr2t::address_of_id:
  {
    const address_of2t &ref2 = to_address_of2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt address_ofval("address_of", thetype);
    address_ofval.copy_to_operands(migrate_expr_back(ref2.ptr_obj));
    return address_ofval;
  }
  case expr2t::byte_extract_id:
  {
    const byte_extract2t &ref2 = to_byte_extract2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt byte_extract(
      (ref2.big_endian) ? "byte_extract_big_endian"
                        : "byte_extract_little_endian",
      thetype);
    byte_extract.copy_to_operands(
      migrate_expr_back(ref2.source_value),
      migrate_expr_back(ref2.source_offset));
    return byte_extract;
  }
  case expr2t::byte_update_id:
  {
    const byte_update2t &ref2 = to_byte_update2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt byte_update(
      (ref2.big_endian) ? "byte_update_big_endian"
                        : "byte_update_little_endian",
      thetype);
    byte_update.copy_to_operands(
      migrate_expr_back(ref2.source_value),
      migrate_expr_back(ref2.source_offset),
      migrate_expr_back(ref2.update_value));
    return byte_update;
  }
  case expr2t::with_id:
  {
    const with2t &ref2 = to_with2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt with("with", thetype);

    exprt memb_name;
    if(is_constant_string2t(ref2.update_field))
    {
      const constant_string2t &string_ref =
        to_constant_string2t(ref2.update_field);
      memb_name = exprt("member_name");
      memb_name.component_name(string_ref.value);
    }
    else
    {
      memb_name = migrate_expr_back(ref2.update_field);
    }

    with.copy_to_operands(
      migrate_expr_back(ref2.source_value),
      memb_name,
      migrate_expr_back(ref2.update_value));
    return with;
  }
  case expr2t::member_id:
  {
    const member2t &ref2 = to_member2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt member("member", thetype);
    member.set("component_name", ref2.member);
    exprt member_name("member_name");
    member.copy_to_operands(migrate_expr_back(ref2.source_value));
    return member;
  }
  case expr2t::index_id:
  {
    const index2t &ref2 = to_index2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt index("index", thetype);
    index.copy_to_operands(
      migrate_expr_back(ref2.source_value), migrate_expr_back(ref2.index));
    return index;
  }
  case expr2t::isnan_id:
  {
    const isnan2t &ref2 = to_isnan2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt isnan("isnan", thetype);
    isnan.copy_to_operands(migrate_expr_back(ref2.value));
    return isnan;
  }
  case expr2t::overflow_id:
  {
    const overflow2t &ref2 = to_overflow2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt theexpr;
    theexpr.type() = thetype;
    if(is_add2t(ref2.operand))
    {
      theexpr.id("overflow-+");
      const add2t &addref = to_add2t(ref2.operand);
      theexpr.copy_to_operands(
        migrate_expr_back(addref.side_1), migrate_expr_back(addref.side_2));
    }
    else if(is_sub2t(ref2.operand))
    {
      theexpr.id("overflow--");
      const sub2t &subref = to_sub2t(ref2.operand);
      theexpr.copy_to_operands(
        migrate_expr_back(subref.side_1), migrate_expr_back(subref.side_2));
    }
    else if(is_mul2t(ref2.operand))
    {
      theexpr.id("overflow-*");
      const mul2t &mulref = to_mul2t(ref2.operand);
      theexpr.copy_to_operands(
        migrate_expr_back(mulref.side_1), migrate_expr_back(mulref.side_2));
    }
    else if(is_div2t(ref2.operand))
    {
      theexpr.id("overflow-/");
      const div2t &divref = to_div2t(ref2.operand);
      theexpr.copy_to_operands(
        migrate_expr_back(divref.side_1), migrate_expr_back(divref.side_2));
    }
    else if(is_modulus2t(ref2.operand))
    {
      theexpr.id("overflow-mod");
      const modulus2t &divref = to_modulus2t(ref2.operand);
      theexpr.copy_to_operands(
        migrate_expr_back(divref.side_1), migrate_expr_back(divref.side_2));
    }
    else if(is_shl2t(ref2.operand))
    {
      theexpr.id("overflow-shl");
      const shl2t &divref = to_shl2t(ref2.operand);
      theexpr.copy_to_operands(
        migrate_expr_back(divref.side_1), migrate_expr_back(divref.side_2));
    }
    else
    {
      assert(0 && "Invalid operand to overflow2t when backmigrating");
    }
    return theexpr;
  }
  case expr2t::overflow_cast_id:
  {
    const overflow_cast2t &ref2 = to_overflow_cast2t(ref);
    char buffer[32];
    snprintf(buffer, 31, "%d", ref2.bits);
    buffer[31] = '\0';

    irep_idt tmp("overflow-typecast-" + std::string(buffer));
    exprt theexpr(tmp);
    typet thetype = migrate_type_back(ref->type);
    theexpr.type() = thetype;
    theexpr.copy_to_operands(migrate_expr_back(ref2.operand));
    return theexpr;
  }
  case expr2t::overflow_neg_id:
  {
    const overflow_neg2t &ref2 = to_overflow_neg2t(ref);
    exprt theexpr("overflow-unary-");
    typet thetype = migrate_type_back(ref->type);
    theexpr.type() = thetype;
    theexpr.copy_to_operands(migrate_expr_back(ref2.operand));
    return theexpr;
  }
  case expr2t::invalid_id:
  {
    typet thetype = migrate_type_back(ref->type);
    const exprt theexpr("invalid", thetype);
    return theexpr;
  }
  case expr2t::unknown_id:
  {
    typet thetype = migrate_type_back(ref->type);
    const exprt theexpr("unknown", thetype);
    return theexpr;
  }
  case expr2t::null_object_id:
  {
    typet thetype = migrate_type_back(ref->type);
    const exprt theexpr("NULL-object", thetype);
    return theexpr;
  }
  case expr2t::dynamic_object_id:
  {
    const dynamic_object2t &ref2 = to_dynamic_object2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt op0 = migrate_expr_back(ref2.instance);
    exprt op1;
    if(ref2.invalid)
      op1 = true_exprt();
    else
      op1 = false_exprt();
    exprt theexpr("dynamic_object", thetype);
    theexpr.copy_to_operands(op0, op1);
    return theexpr;
  }
  case expr2t::dereference_id:
  {
    const dereference2t &ref2 = to_dereference2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt op0 = migrate_expr_back(ref2.value);
    exprt theexpr("dereference", thetype);
    theexpr.copy_to_operands(op0);
    return theexpr;
  }
  case expr2t::valid_object_id:
  {
    const valid_object2t &ref2 = to_valid_object2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt op0 = migrate_expr_back(ref2.value);
    exprt theexpr("valid_object", thetype);
    theexpr.copy_to_operands(op0);
    return theexpr;
  }
  case expr2t::deallocated_obj_id:
  {
    const deallocated_obj2t &ref2 = to_deallocated_obj2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt op0 = migrate_expr_back(ref2.value);
    exprt theexpr("deallocated_object", thetype);
    theexpr.copy_to_operands(op0);
    return theexpr;
  }
  case expr2t::dynamic_size_id:
  {
    const dynamic_size2t &ref2 = to_dynamic_size2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt op0 = migrate_expr_back(ref2.value);
    exprt theexpr("dynamic_size", thetype);
    theexpr.copy_to_operands(op0);
    return theexpr;
  }
  case expr2t::sideeffect_id:
  {
    const sideeffect2t &ref2 = to_sideeffect2t(ref);
    typet thetype = migrate_type_back(ref->type);
    exprt theexpr("sideeffect", thetype);
    typet cmttype;
    exprt size;

    if(!is_nil_type(ref2.alloctype))
      cmttype = migrate_type_back(ref2.alloctype);

    if(!is_nil_expr(ref2.size))
      size = migrate_expr_back(ref2.size);

    if(ref2.kind == sideeffect2t::function_call)
    {
      // "Operand" is 1st op,
      exprt operand = migrate_expr_back(ref2.operand);
      // 2nd op is "arguments".
      exprt args("arguments");
      for(const auto &argument : ref2.arguments)
        args.copy_to_operands(migrate_expr_back(argument));
      theexpr.copy_to_operands(operand, args);
    }
    else if(ref2.kind == sideeffect2t::nondet)
    {
      ; // Do nothing
    }
    else
    {
      exprt operand = migrate_expr_back(ref2.operand);
      theexpr.copy_to_operands(operand);
    }

    theexpr.cmt_type(cmttype);
    theexpr.cmt_size(size);

    switch(ref2.kind)
    {
    case sideeffect2t::malloc:
      theexpr.statement("malloc");
      break;
    case sideeffect2t::realloc:
      theexpr.statement("realloc");
      break;
    case sideeffect2t::alloca:
      theexpr.statement("alloca");
      break;
    case sideeffect2t::cpp_new:
      theexpr.statement("cpp_new");
      break;
    case sideeffect2t::cpp_new_arr:
      theexpr.statement("cpp_new[]");
      break;
    case sideeffect2t::nondet:
      theexpr.statement("nondet");
      break;
    case sideeffect2t::va_arg:
      theexpr.statement("va_arg");
      break;
    case sideeffect2t::function_call:
      theexpr.statement("function_call");
      break;
    case sideeffect2t::preincrement:
      theexpr.statement("preincrement");
      break;
    case sideeffect2t::postincrement:
      theexpr.statement("postincrement");
      break;
    case sideeffect2t::predecrement:
      theexpr.statement("predecrement");
      break;
    case sideeffect2t::postdecrement:
      theexpr.statement("postdecrement");
      break;
    default:
      assert(0 && "Unexpected side effect type when back-converting");
    }

    return theexpr;
  }
  case expr2t::code_assign_id:
  {
    const code_assign2t &ref2 = to_code_assign2t(ref);
    exprt codeexpr("code", code_typet());
    codeexpr.statement(irep_idt("assign"));
    exprt op0 = migrate_expr_back(ref2.target);
    exprt op1 = migrate_expr_back(ref2.source);
    codeexpr.copy_to_operands(op0, op1);
    return codeexpr;
  }
  case expr2t::code_decl_id:
  {
    const code_decl2t &ref2 = to_code_decl2t(ref);
    exprt codeexpr("code", code_typet());
    codeexpr.statement(irep_idt("decl"));
    typet thetype = migrate_type_back(ref2.type);
    exprt symbol = symbol_exprt(ref2.value, thetype);
    codeexpr.copy_to_operands(symbol);
    return codeexpr;
  }
  case expr2t::code_dead_id:
  {
    const code_dead2t &ref2 = to_code_dead2t(ref);
    exprt codeexpr("code", code_typet());
    codeexpr.statement(irep_idt("dead"));
    typet thetype = migrate_type_back(ref2.type);
    exprt symbol = symbol_exprt(ref2.value, thetype);
    codeexpr.copy_to_operands(symbol);
    return codeexpr;
  }
  case expr2t::code_printf_id:
  {
    const code_printf2t &ref2 = to_code_printf2t(ref);
    exprt codeexpr("code", code_typet());
    codeexpr.statement(irep_idt("printf"));
    for(auto const &it : ref2.operands)
      codeexpr.operands().push_back(migrate_expr_back(it));
    return codeexpr;
  }
  case expr2t::code_expression_id:
  {
    const code_expression2t &ref2 = to_code_expression2t(ref);
    exprt codeexpr("code", code_typet());
    codeexpr.statement(irep_idt("expression"));
    exprt op0 = migrate_expr_back(ref2.operand);
    codeexpr.copy_to_operands(op0);
    return codeexpr;
  }
  case expr2t::code_return_id:
  {
    const code_return2t &ref2 = to_code_return2t(ref);
    exprt codeexpr("code", code_typet());
    codeexpr.statement(irep_idt("return"));
    exprt op0 = migrate_expr_back(ref2.operand);
    codeexpr.copy_to_operands(op0);
    return codeexpr;
  }
  case expr2t::code_skip_id:
  {
    exprt codeexpr("code", code_typet());
    codeexpr.statement("skip");
    return codeexpr;
  }
  case expr2t::code_free_id:
  {
    const code_free2t &ref2 = to_code_free2t(ref);
    exprt codeexpr("code", code_typet());
    codeexpr.statement(irep_idt("free"));
    exprt op0 = migrate_expr_back(ref2.operand);
    codeexpr.copy_to_operands(op0);
    return codeexpr;
  }
  case expr2t::object_descriptor_id:
  {
    const object_descriptor2t &ref2 = to_object_descriptor2t(ref);
    typet thetype = migrate_type_back(ref2.type);
    exprt obj("object_descriptor", thetype);
    exprt op0 = migrate_expr_back(ref2.object);
    exprt op1 = migrate_expr_back(ref2.offset);
    obj.copy_to_operands(op0, op1);
    return obj;
  }
  case expr2t::code_function_call_id:
  {
    const code_function_call2t &ref2 = to_code_function_call2t(ref);
    exprt codeexpr("code", code_typet());
    codeexpr.statement(irep_idt("function_call"));
    exprt op0 = migrate_expr_back(ref2.ret);
    exprt op1 = migrate_expr_back(ref2.function);
    exprt op2("arguments");
    codeexpr.copy_to_operands(op0, op1, op2);
    exprt &args = codeexpr.op2();
    for(auto const &it : ref2.operands)
      args.operands().push_back(migrate_expr_back(it));
    return codeexpr;
  }
  case expr2t::code_comma_id:
  {
    const code_comma2t &ref2 = to_code_comma2t(ref);
    exprt codeexpr("comma", migrate_type_back(ref2.type));
    codeexpr.copy_to_operands(
      migrate_expr_back(ref2.side_1), migrate_expr_back(ref2.side_2));
    return codeexpr;
  }
  case expr2t::invalid_pointer_id:
  {
    const invalid_pointer2t &ref2 = to_invalid_pointer2t(ref);
    exprt theexpr("invalid-pointer", bool_typet());
    theexpr.copy_to_operands(migrate_expr_back(ref2.ptr_obj));
    return theexpr;
  }
  case expr2t::code_goto_id:
  {
    const code_goto2t &ref2 = to_code_goto2t(ref);
    exprt codeexpr("code", code_typet());
    codeexpr.set("destination", ref2.target);
    return codeexpr;
  }
  case expr2t::code_asm_id:
  {
    const code_asm2t &ref2 = to_code_asm2t(ref);
    exprt codeexpr("code", migrate_type_back(ref2.type));
    codeexpr.statement("asm");
    // Don't actually set a piece of assembly as the operand here; it serves
    // no purpose.
    codeexpr.operands().resize(1);
    codeexpr.op0() = exprt("string-constant");
    return codeexpr;
  }
  case expr2t::code_cpp_del_array_id:
  {
    const code_cpp_del_array2t &ref2 = to_code_cpp_del_array2t(ref);
    exprt codeexpr("cpp_delete[]", typet());
    codeexpr.copy_to_operands(migrate_expr_back(ref2.operand));
    return codeexpr;
  }
  case expr2t::code_cpp_delete_id:
  {
    const code_cpp_delete2t &ref2 = to_code_cpp_delete2t(ref);
    exprt codeexpr("cpp_delete", typet());
    codeexpr.copy_to_operands(migrate_expr_back(ref2.operand));
    return codeexpr;
  }
  case expr2t::code_cpp_throw_id:
  {
    const code_cpp_throw2t &ref2 = to_code_cpp_throw2t(ref);
    exprt codeexpr("cpp-throw");
    irept::subt &exceptions_thrown = codeexpr.add("exception_list").get_sub();

    for(auto const &it : ref2.exception_list)
      exceptions_thrown.emplace_back(it);

    codeexpr.copy_to_operands(migrate_expr_back(ref2.operand));
    return codeexpr;
  }
  case expr2t::isinf_id:
  {
    const isinf2t &ref2 = to_isinf2t(ref);
    exprt back("isinf", bool_typet());
    back.copy_to_operands(migrate_expr_back(ref2.value));
    return back;
  }
  case expr2t::isnormal_id:
  {
    const isnormal2t &ref2 = to_isnormal2t(ref);
    exprt back("isnormal", bool_typet());
    back.copy_to_operands(migrate_expr_back(ref2.value));
    return back;
  }
  case expr2t::isfinite_id:
  {
    const isfinite2t &ref2 = to_isfinite2t(ref);
    exprt back("isfinite", bool_typet());
    back.copy_to_operands(migrate_expr_back(ref2.value));
    return back;
  }
  case expr2t::signbit_id:
  {
    const signbit2t &ref2 = to_signbit2t(ref);
    exprt back("signbit", bool_typet());
    back.copy_to_operands(migrate_expr_back(ref2.operand));
    return back;
  }
  case expr2t::popcount_id:
  {
    const popcount2t &ref2 = to_popcount2t(ref);
    exprt back("popcount", migrate_type_back(ref->type));
    back.copy_to_operands(migrate_expr_back(ref2.operand));
    return back;
  }
  case expr2t::bswap_id:
  {
    const bswap2t &ref2 = to_bswap2t(ref);
    exprt back("bswap", migrate_type_back(ref->type));
    back.copy_to_operands(migrate_expr_back(ref2.value));
    return back;
  }
  case expr2t::concat_id:
  {
    const concat2t &ref2 = to_concat2t(ref);
    exprt back("concat", migrate_type_back(ref2.type));
    back.copy_to_operands(migrate_expr_back(ref2.side_1));
    back.copy_to_operands(migrate_expr_back(ref2.side_2));
    return back;
  }
  case expr2t::extract_id:
  {
    const extract2t &ref2 = to_extract2t(ref);
    exprt back("extract", migrate_type_back(ref2.type));
    back.copy_to_operands(migrate_expr_back(ref2.from));

    back.set("upper", irep_idt(std::to_string(ref2.upper)));
    back.set("lower", irep_idt(std::to_string(ref2.lower)));
    return back;
  }
  case expr2t::bitcast_id:
  {
    const bitcast2t &ref2 = to_bitcast2t(ref);
    exprt back("bitcast", migrate_type_back(ref2.type));
    back.copy_to_operands(migrate_expr_back(ref2.from));
    back.set("rounding_mode", migrate_expr_back(ref2.rounding_mode));
    return back;
  }
  default:
    std::cerr << "Unrecognized expr in migrate_expr_back" << std::endl;
    abort();
  }
}
