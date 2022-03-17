/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <clang-c-frontend/expr2ccode.h>
#include <util/arith_tools.h>
#include <util/c_misc.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/fixedbv.h>
#include <util/i2string.h>
#include <util/ieee_float.h>
#include <util/prefix.h>
#include <util/std_code.h>
#include <util/std_types.h>

#include <util/type_byte_size.h>

#include <pointer-analysis/dereference.h>

#include <algorithm>
#include <regex>

#include <iostream>

std::string
expr2ccode(const exprt &expr, const namespacet &ns, bool fullname)
{
  std::string code;
  expr2ccodet expr2ccode(ns, fullname);
  expr2ccode.get_shorthands(expr);
  return expr2ccode.convert(expr);
}

std::string
type2ccode(const typet &type, const namespacet &ns, bool fullname)
{
  expr2ccodet expr2ccode(ns, fullname);
  return expr2ccode.convert(type);
}

std::string expr2ccodet::convert(const typet &src)
{
  return convert_rec(src, c_qualifierst(), "");
}

std::string expr2ccodet::convert(const exprt &src)
{
  unsigned precedence;
  return convert(src, precedence);
}

std::string expr2ccodet::convert_rec(
  const typet &src,
  const c_qualifierst &qualifiers,
  const std::string &declarator)
{
  c_qualifierst new_qualifiers(qualifiers);
  new_qualifiers.read(src);

  std::string q = new_qualifiers.as_string();

  std::string d = declarator == "" ? declarator : " " + declarator;

  if(src.is_bool())
  {
    return q + "_Bool" + d;
  }
  if(src.id() == "empty")
  {
    return q + "void" + d;
  }
  else if(src.id() == "signedbv" || src.id() == "unsignedbv")
  {
    BigInt width = string2integer(src.width().as_string());

    bool is_signed = src.id() == "signedbv";
    std::string sign_str = is_signed ? "signed " : "unsigned ";

    if(width == config.ansi_c.int_width)
      return q + sign_str + "int" + d;

    if(width == config.ansi_c.long_int_width)
      return q + sign_str + "long int" + d;

    if(width == config.ansi_c.char_width)
      return q + sign_str + "char" + d;

    if(width == config.ansi_c.short_int_width)
      return q + sign_str + "short int" + d;

    if(width == config.ansi_c.long_long_int_width)
      return q + sign_str + "long long int" + d;

    return q + sign_str + "_ExtInt(" + std::to_string(width.to_uint64()) + ")" +
           d;
  }
  else if(src.id() == "floatbv" || src.id() == "fixedbv")
  {
    BigInt width = string2integer(src.width().as_string());

    if(width == config.ansi_c.single_width)
      return q + "float" + d;
    if(width == config.ansi_c.double_width)
      return q + "double" + d;
    else if(width == config.ansi_c.long_double_width)
      return q + "long double" + d;
  }
  else if(src.id() == "struct")
  {
    const struct_typet &struct_type = to_struct_type(src);

    std::string dest = q;

    const irep_idt &tag = struct_type.tag().as_string();
    if(tag != "")
      dest += " " + id2string(tag);

    dest += " {";

    for(const auto &component : struct_type.components())
    {
      dest += ' ';
      dest += convert_rec(
        component.type(), c_qualifierst(), id2string(component.get_name()));
      dest += ';';
    }

    dest += " }";
    dest += declarator;
    return dest;
  }
  else if(src.id() == "incomplete_struct")
  {
    std::string dest = q + "struct";
    const std::string &tag = src.tag().as_string();
    if(tag != "")
      dest += " " + tag;
    dest += d;
    return dest;
  }
  else if(src.id() == "union")
  {
    const union_typet &union_type = to_union_type(src);

    std::string dest = q;

    const irep_idt &tag = union_type.tag().as_string();
    if(tag != "")
      dest += " " + id2string(tag);
    dest += " {";
    for(auto const &it : union_type.components())
    {
      dest += ' ';
      dest += convert_rec(it.type(), c_qualifierst(), id2string(it.get_name()));
      dest += ';';
    }
    dest += " }";
    dest += d;
    return dest;
  }
  else if(src.id() == "c_enum" || src.id() == "incomplete_c_enum")
  {
    std::string result = q + "enum";
    if(src.name() != "")
      result += " " + src.tag().as_string();
    result += d;
    return result;
  }
  else if(src.id() == "pointer")
  {
    if(src.subtype().is_code())
    {
      const typet &return_type = (typet &)src.subtype().return_type();

      std::string dest = q + convert(return_type);

      // function "name"
      dest += " (*)";

      // arguments
      dest += "(";
      const irept &arguments = src.subtype().arguments();

      forall_irep(it, arguments.get_sub())
      {
        const typet &argument_type = ((exprt &)*it).type();

        if(it != arguments.get_sub().begin())
          dest += ", ";

        dest += convert(argument_type);
      }

      dest += ")";

      return dest;
    }

    std::string tmp = convert(src.subtype());

    if(q == "")
      return tmp + " *" + d;

    return q + " (" + tmp + " *)" + d;
  }
  else if(src.is_array())
  {
    std::string size_string =
      convert(static_cast<const exprt &>(src.size_irep()));
    return convert(src.subtype()) + d + " [" + size_string + "]";
  }
  else if(src.id() == "incomplete_array")
  {
    return convert(src.subtype()) + " []";
  }
  else if(src.id() == "symbol")
  {
    const typet &followed = ns.follow(src);
    if(followed.id() == "struct")
    {
      std::string dest = q;
      const std::string &tag = followed.tag().as_string();
      if(tag != "")
        dest += " " + tag;
      dest += d;
      return dest;
    }

    if(followed.id() == "union")
    {
      std::string dest = q;
      const std::string &tag = followed.tag().as_string();
      if(tag != "")
        dest += " " + tag;
      dest += d;
      return dest;
    }

    return convert_rec(ns.follow(src), new_qualifiers, declarator);
  }
  else if(src.is_code())
  {
    const typet &return_type = (typet &)src.return_type();

    std::string dest = convert(return_type) + " ";

    dest += "(";
    const irept &arguments = src.arguments();

    forall_irep(it, arguments.get_sub())
    {
      const typet &argument_type = ((exprt &)*it).type();

      if(it != arguments.get_sub().begin())
        dest += ", ";

      dest += convert(argument_type);
    }

    dest += ")";
    return dest;
  }

  unsigned precedence;
  return convert_norep((exprt &)src, precedence);
}

std::string expr2ccodet::convert_code_printf (const codet &src, unsigned indent)
{
  std::string dest = indent_str(indent) + "printf(";

  forall_operands(it, src)
  {
    unsigned p;
    std::string arg_str = convert(*it, p);

    if(it != src.operands().begin())
      dest += ", ";
    // TODO: [add] brackets, if necessary, depending on p
    dest += arg_str;
  }

  dest += ");";

  return dest;
}

std::string expr2ccodet::convert_symbol(const exprt &src, unsigned &)
{
  const irep_idt &id = src.identifier();
  std::string dest;

  if(!fullname && ns_collision.find(id) == ns_collision.end())
    dest = id_shorthand(src);
  else
    dest = id2string(id);

  if(src.id() == "next_symbol")
    dest = "NEXT(" + dest + ")";

  //std::replace(dest.begin(), dest.end(), '$', '_');
  //std::replace(dest.begin(), dest.end(), '?', '_');
  //std::replace(dest.begin(), dest.end(), '!', '_');
  //std::replace(dest.begin(), dest.end(), '&', '_');
  //std::replace(dest.begin(), dest.end(), '#', '_');

  dest = convert_from_ssa_form(dest);

  return dest;
}

std::string expr2ccodet::convert(const exprt &src, unsigned &precedence)
{
  precedence = 16;

  if(src.id() == "+")
    return convert_binary(src, "+", precedence = 12, false);

  if(src.id() == "-")
  {
    if(src.operands().size() == 1)
      return convert_norep(src, precedence);

    return convert_binary(src, "-", precedence = 12, true);
  }

  else if(src.id() == "unary-")
  {
    if(src.operands().size() != 1)
      return convert_norep(src, precedence);

    return convert_unary(src, "-", precedence = 15);
  }

  else if(src.id() == "unary+")
  {
    if(src.operands().size() != 1)
      return convert_norep(src, precedence);

    return convert_unary(src, "+", precedence = 15);
  }

  else if(src.id() == "invalid-pointer")
  {
    return convert_function(src, "INVALID_POINTER", precedence = 15);
  }

  else if(src.id() == "invalid-object")
  {
    return "invalid_object";
  }

  else if(src.id() == "NULL-object")
  {
    return "0";
  }

  else if(src.id() == "infinity")
  {
    return convert_function(src, "INFINITY", precedence = 15);
  }

  else if(src.id() == "builtin-function")
  {
    return src.identifier().as_string();
  }

  else if(src.id() == "pointer_object")
  {
    return convert_function(src, "__ESBMC_POINTER_OBJECT", precedence = 15);
  }

  else if(src.id() == "object_value")
  {
    return convert_function(src, "OBJECT_VALUE", precedence = 15);
  }

  else if(src.id() == "pointer_object_has_type")
  {
    return convert_pointer_object_has_type(src, precedence = 15);
  }

  else if(src.id() == "array_of")
  {
    return convert_array_of(src, precedence = 15);
  }

  else if(src.id() == "pointer_offset")
  {
    return convert_function(src, "POINTER_OFFSET", precedence = 15);
  }

  else if(src.id() == "pointer_base")
  {
    return convert_function(src, "POINTER_BASE", precedence = 15);
  }

  else if(src.id() == "pointer_cons")
  {
    return convert_function(src, "POINTER_CONS", precedence = 15);
  }

  else if(src.id() == "same-object")
  {
    return convert_same_object(src, precedence = 15);
  }

  else if(src.id() == "valid_object")
  {
    return convert_function(src, "VALID_OBJECT", precedence = 15);
  }

  else if(src.id() == "deallocated_object" || src.id() == "memory-leak")
  {
    return convert_function(src, "DEALLOCATED_OBJECT", precedence = 15);
  }

  else if(src.id() == "dynamic_object")
  {
    return convert_function(src, "DYNAMIC_OBJECT", precedence = 15);
  }

  else if(src.id() == "is_dynamic_object")
  {
    return convert_function(src, "IS_DYNAMIC_OBJECT", precedence = 15);
  }

  else if(src.id() == "dynamic_size")
  {
    return convert_function(src, "DYNAMIC_SIZE", precedence = 15);
  }

  else if(src.id() == "dynamic_type")
  {
    return convert_function(src, "DYNAMIC_TYPE", precedence = 15);
  }

  else if(src.id() == "pointer_offset")
  {
    return convert_function(src, "POINTER_OFFSET", precedence = 15);
  }

  else if(src.id() == "isnan")
  {
    return convert_function(src, "isnan", precedence = 15);
  }

  else if(src.id() == "isfinite")
  {
    return convert_function(src, "isfinite", precedence = 15);
  }

  else if(src.id() == "isinf")
  {
    return convert_function(src, "isinf", precedence = 15);
  }

  else if(src.id() == "isnormal")
  {
    return convert_function(src, "isnormal", precedence = 15);
  }

  else if(src.id() == "signbit")
  {
    return convert_function(src, "signbit", precedence = 15);
  }

  else if(src.id() == "nearbyint")
  {
    return convert_function(src, "nearbyint", precedence = 15);
  }

  else if(src.id() == "popcount")
  {
    return convert_function(src, "popcount", precedence = 15);
  }

  else if(src.id() == "bswap")
  {
    return convert_function(src, "bswap", precedence = 15);
  }

  else if(src.id() == "builtin_va_arg")
  {
    return convert_function(src, "builtin_va_arg", precedence = 15);
  }

  else if(has_prefix(src.id_string(), "byte_extract"))
  {
    return convert_byte_extract(src, precedence = 15);
  }

  else if(has_prefix(src.id_string(), "byte_update"))
  {
    return convert_byte_update(src, precedence = 15);
  }

  else if(src.is_address_of())
  {
    if(src.operands().size() != 1)
      return convert_norep(src, precedence);
    if(src.op0().id() == "label")
      return "&&" + src.op0().get_string("identifier");
    else
      return convert_unary(src, "&", precedence = 15);
  }

  else if(src.id() == "dereference")
  {
    if(src.operands().size() != 1)
      return convert_norep(src, precedence);

    return convert_unary(src, "*", precedence = 15);
  }

  else if(src.id() == "index")
    return convert_index(src, precedence = 16);

  else if(src.id() == "member")
    return convert_member(src, precedence = 16);

  else if(src.id() == "array-member-value")
    return convert_array_member_value(src, precedence = 16);

  else if(src.id() == "struct-member-value")
    return convert_struct_member_value(src, precedence = 16);

  else if(src.id() == "sideeffect")
  {
    const irep_idt &statement = src.statement();
    if(statement == "preincrement")
      return convert_unary(src, "++", precedence = 15);
    if(statement == "predecrement")
      return convert_unary(src, "--", precedence = 15);
    else if(statement == "postincrement")
      return convert_unary_post(src, "++", precedence = 16);
    else if(statement == "postdecrement")
      return convert_unary_post(src, "--", precedence = 16);
    else if(statement == "assign+")
      return convert_binary(src, "+=", precedence = 2, true);
    else if(statement == "assign-")
      return convert_binary(src, "-=", precedence = 2, true);
    else if(statement == "assign*")
      return convert_binary(src, "*=", precedence = 2, true);
    else if(statement == "assign_div")
      return convert_binary(src, "/=", precedence = 2, true);
    else if(statement == "assign_mod")
      return convert_binary(src, "%=", precedence = 2, true);
    else if(statement == "assign_shl")
      return convert_binary(src, "<<=", precedence = 2, true);
    else if(statement == "assign_ashr")
      return convert_binary(src, ">>=", precedence = 2, true);
    else if(statement == "assign_bitand")
      return convert_binary(src, "&=", precedence = 2, true);
    else if(statement == "assign_bitxor")
      return convert_binary(src, "^=", precedence = 2, true);
    else if(statement == "assign_bitor")
      return convert_binary(src, "|=", precedence = 2, true);
    else if(statement == "assign")
      return convert_binary(src, "=", precedence = 2, true);
    else if(statement == "function_call")
      return convert_function_call(src, precedence);
    else if(statement == "malloc")
      return convert_malloc(src, precedence = 15);
    else if(statement == "realloc")
      return convert_realloc(src, precedence = 15);
    else if(statement == "alloca")
      return convert_alloca(src, precedence = 15);
    else if(statement == "printf")
      return convert_function(src, "PRINTF", precedence = 15);
    else if(statement == "nondet")
      return convert_nondet(src, precedence = 15);
    else if(statement == "statement_expression")
      return convert_statement_expression(src, precedence = 15);
    else if(statement == "va_arg")
      return convert_function(src, "va_arg", precedence = 15);
    else
      return convert_norep(src, precedence);
  }

  else if(src.id() == "not")
    return convert_unary(src, "!", precedence = 15);

  else if(src.id() == "bitnot")
    return convert_unary(src, "~", precedence = 15);

  else if(src.id() == "*")
    return convert_binary(src, src.id_string(), precedence = 13, false);

  else if(src.id() == "/")
    return convert_binary(src, src.id_string(), precedence = 13, true);

  else if(src.id() == "mod")
    return convert_binary(src, "%", precedence = 13, true);

  else if(src.id() == "shl")
    return convert_binary(src, "<<", precedence = 11, true);

  else if(src.id() == "ashr" || src.id() == "lshr")
    return convert_binary(src, ">>", precedence = 11, true);

  else if(
    src.id() == "<" || src.id() == ">" || src.id() == "<=" || src.id() == ">=")
    return convert_binary(src, src.id_string(), precedence = 10, true);

  else if(src.id() == "notequal")
    return convert_binary(src, "!=", precedence = 9, true);

  else if(src.id() == "=")
    return convert_binary(src, "==", precedence = 9, true);

  else if(src.id() == "ieee_add")
    return convert_function(src, "IEEE_ADD", precedence = 15);

  else if(src.id() == "ieee_sub")
    return convert_function(src, "IEEE_SUB", precedence = 15);

  else if(src.id() == "ieee_mul")
    return convert_function(src, "IEEE_MUL", precedence = 15);

  else if(src.id() == "ieee_div")
    return convert_function(src, "IEEE_DIV", precedence = 15);

  else if(src.id() == "width")
    return convert_function(src, "WIDTH", precedence = 15);

  else if(src.id() == "byte_update_little_endian")
    return convert_function(src, "BYTE_UPDATE_LITTLE_ENDIAN", precedence = 15);

  else if(src.id() == "byte_update_big_endian")
    return convert_function(src, "BYTE_UPDATE_BIG_ENDIAN", precedence = 15);

  else if(src.id() == "abs")
    return convert_function(src, "abs", precedence = 15);

  else if(src.id() == "bitand")
    return convert_binary(src, "&", precedence = 8, false);

  else if(src.id() == "bitxor")
    return convert_binary(src, "^", precedence = 7, false);

  else if(src.id() == "bitor")
    return convert_binary(src, "|", precedence = 6, false);

  else if(src.is_and())
    return convert_binary(src, "&&", precedence = 5, false);

  else if(src.id() == "or")
    return convert_binary(src, "||", precedence = 4, false);

  else if(src.id() == "=>")
    return convert_binary(src, "=>", precedence = 3, true);

  else if(src.id() == "if")
    return convert_trinary(src, "?", ":", precedence = 3);

  else if(src.id() == "forall")
    return convert_quantifier(src, "FORALL", precedence = 2);

  else if(src.id() == "exists")
    return convert_quantifier(src, "EXISTS", precedence = 2);

  else if(src.id() == "with")
    return convert_with(src, precedence = 2);

  else if(src.id() == "symbol")
    return convert_symbol(src, precedence);

  else if(src.id() == "next_symbol")
    return convert_symbol(src, precedence);

  else if(src.id() == "nondet_symbol")
    return convert_nondet_symbol(src, precedence);

  else if(src.id() == "predicate_symbol")
    return convert_predicate_symbol(src, precedence);

  else if(src.id() == "predicate_next_symbol")
    return convert_predicate_next_symbol(src, precedence);

  else if(src.id() == "quantified_symbol")
    return convert_quantified_symbol(src, precedence);

  else if(src.id() == "nondet_bool")
    return convert_nondet_bool(src, precedence);

  else if(src.id() == "object_descriptor")
    return convert_object_descriptor(src, precedence);

  else if(src.id() == "Hoare")
    return convert_Hoare(src);

  else if(src.is_code())
    return convert_code(to_code(src));

  else if(src.id() == "constant")
    return convert_constant(src, precedence);

  else if(src.id() == "string-constant")
    return convert_constant(src, precedence);

  else if(src.id() == "struct")
    return convert_struct(src, precedence);

  else if(src.id() == "union")
    return convert_union(src, precedence);

  else if(src.is_array())
    return convert_array(src, precedence);

  else if(src.id() == "array-list")
    return convert_array_list(src, precedence);

  else if(src.id() == "typecast")
    return convert_typecast(src, precedence);

  else if(src.id() == "bitcast")
    return convert_bitcast(src, precedence);

  else if(src.id() == "implicit_address_of")
    return convert_implicit_address_of(src, precedence);

  else if(src.id() == "implicit_dereference")
    return convert_function(src, "IMPLICIT_DEREFERENCE", precedence = 15);

  else if(src.id() == "comma")
    return convert_binary(src, ", ", precedence = 1, false);

  else if(src.id() == "cond")
    return convert_cond(src, precedence);

  else if(std::string(src.id_string(), 0, 9) == "overflow-")
    return convert_overflow(src, precedence);

  else if(src.id() == "unknown")
    return "*";

  else if(src.id() == "invalid")
    return "#";

  else if(src.id() == "extractbit")
    return convert_extractbit(src, precedence);

  else if(src.id() == "sizeof")
    return convert_sizeof(src, precedence);

  else if(src.id() == "concat")
    return convert_function(src, "CONCAT", precedence = 15);

  else if(src.id() == "extract")
    return convert_extract(src);

  // no C language expression for internal representation
  return convert_norep(src, precedence);
}

std::string expr2ccodet::convert_from_ssa_form(const std::string symbol)
{
  std::string new_symbol = symbol;
  std::regex symbol_ssa_addon("(\\?[0-9]+![0-9]+)|(&[0-9]+#[0-9]+)");
  new_symbol = std::regex_replace(new_symbol, symbol_ssa_addon, "");
  return new_symbol;
}

std::string expr2ccodet::convert_same_object(const exprt &src, unsigned &precedence)
{
  assert(src.operands().size() == 2);
  expr2tc new_src;
  migrate_expr(src, new_src);

  assert(is_same_object2t(new_src));
  same_object2tc same = to_same_object2t(new_src);

  assert(is_pointer_type(same->side_1->type) && 
          is_pointer_type(same->side_2->type));
  pointer_type2tc ptr_type = to_pointer_type(same->side_1->type);
  unsigned int ptr_subtype_size = type_byte_size(ptr_type->subtype).to_uint64();
  
  if(is_address_of2t(same->side_2))
  {
    address_of2tc addr = to_address_of2t(same->side_2);
    if(is_array_type(addr->ptr_obj->type))
    {
      array_type2tc arr_type = to_array_type(addr->ptr_obj->type); 
      unsigned int arr_subtype_size = type_byte_size(arr_type->subtype).to_uint64();
      // This arr_size is equal to the number of elements of the given array
      expr2tc arr_size = arr_type->array_size;
      greaterthan2tc gt(addr->ptr_obj, same->side_1);
      greaterthan2tc gt2(add2tc(same->side_1->type, same->side_1, gen_ulong(1)), 
                          add2tc(arr_type->subtype, addr->ptr_obj, arr_size));
      or2tc in_bounds(gt, gt2);
      simplify(in_bounds);
      return convert(migrate_expr_back(in_bounds), precedence);
    }
  }

  equality2tc eq(same->side_1, same->side_2);
  return convert(migrate_expr_back(not2tc(eq)), precedence);
//  return "/* unsupported case of SAME-OBJECT:\n" + same->side_2->pretty() + "*/";
}

std::string expr2ccodet::convert_malloc(const exprt &src, unsigned &precedence)
{
  //std::cerr << "// converting malloc = " << src << "\n";
  if(src.operands().size() != 1)
    return convert_norep(src, precedence);

  unsigned p0;
  std::string op0 = convert(src.op0(), p0);

  std::string dest = "malloc";
  dest += '(';
  dest += op0;
  dest += ')';

  return dest;
}
