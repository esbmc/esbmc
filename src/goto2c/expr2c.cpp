#include <goto2c/expr2c.h>
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
#include <util/base_type.h>
#include <util/type_byte_size.h>
#include <irep2/irep2_utils.h>
#include <algorithm>
#include <regex>

std::string expr2ct::get_name_shorthand(std::string fullname)
{
  std::string shorthand = fullname;

  std::string::size_type pos = shorthand.rfind("@");
  if (pos != std::string::npos)
    shorthand.erase(0, pos + 1);

  return shorthand;
}

bool expr2ct::is_padding(std::string tag)
{
  return has_prefix(tag, "anon_pad$") || has_prefix(tag, "$pad") ||
         has_prefix(tag, "anon_bit_field_pad$");
}

bool expr2ct::is_anonymous_member(std::string tag)
{
  return has_prefix(tag, "anon$");
}

// The following is a simple workaround to determine when a struct/union
// type is defined using the "typedef" keyword since it changes
// the type declaration syntax. This information is reflected only in the
// type name in ESBMC irep. (Currently this info is not carried over from the
// Clang AST, and ideally we should introduce a separate
// attribute for this field.)
bool expr2ct::is_typedef_struct_union(std::string tag)
{
  std::smatch m;
  return !(
    std::regex_search(id2string(tag), m, std::regex("union[[:space:]]+.*")) ||
    std::regex_search(id2string(tag), m, std::regex("struct[[:space:]]+.*")));
}

std::string expr2c(const exprt &expr, const namespacet &ns, unsigned flags)
{
  expr2ct expr2c(ns, flags);
  expr2c.get_shorthands(expr);
  return expr2c.convert(expr);
}

std::string type2c(const typet &type, const namespacet &ns, unsigned flags)
{
  expr2ct expr2c(ns, flags);
  return expr2c.convert(type);
}

std::string typedef2c(const typet &type, const namespacet &ns, unsigned flags)
{
  if (type.id() == "struct" || type.id() == "union")
    return expr2ct(ns, flags).convert_struct_union_typedef(
      to_struct_union_type(type));

  return type2c(type, ns, flags);
}

std::string expr2ct::convert(const typet &src)
{
  return convert_rec(src, c_qualifierst(), "");
}

std::string expr2ct::convert(const exprt &src)
{
  unsigned precedence;
  return convert(src, precedence);
}

std::string expr2ct::convert_struct_union_typedef(const struct_union_typet &src)
{
  assert(src.id() == "struct" || src.id() == "union");
  std::string tag = src.tag().as_string();
  std::string dest = "";

  // The following is a simple workaround to determine when a struct
  // type is defined using the "typedef" keyword since it changes
  // the type declaration syntax. This information is reflected only in the
  // type name in ESBMC irep. (Currently this info is not carried over from the
  // Clang AST, and ideally we should introduce a separate attribute for
  // this field.)
  if (is_typedef_struct_union(tag))
    dest += "typedef " + src.id().as_string();
  else
    dest += tag;

  dest += " {\n";
  for (const auto &component : src.components())
  {
    std::string comp_name = component.get_name().as_string();

    // Do not output any padding members
    if (is_padding(comp_name))
      continue;

    dest += " ";
    dest += convert_rec(component.type(), c_qualifierst(), comp_name);
    dest += ";\n";
  }
  dest += "}";

  if (is_typedef_struct_union(tag))
    dest += " " + tag;

  return dest;
}

std::string expr2ct::convert_rec(
  const typet &src,
  const c_qualifierst &qualifiers,
  const std::string &declarator)
{
  c_qualifierst new_qualifiers(qualifiers);
  new_qualifiers.read(src);

  // Constructing qualifiers that go before the type
  std::string q = "";
  if (new_qualifiers.is_constant)
    q += "const ";
  if (new_qualifiers.is_volatile)
    q += "volatile ";

  // The "__restrict" qualifier is used with pointers and goes
  // before the pointer name
  std::string rst_q = "";
  if (new_qualifiers.is_restricted)
    rst_q += "__restrict ";

  std::string d = declarator == "" ? declarator : " " + declarator;

  if (src.is_bool())
  {
    return q + "_Bool" + d;
  }
  if (src.id() == "empty")
  {
    return q + "void" + d;
  }
  else if (src.id() == "signedbv" || src.id() == "unsignedbv")
  {
    BigInt width = string2integer(src.width().as_string());

    // Inferring the type "name" from its width may result in
    // an incorrect/imprecise translation as
    // there is no way of distinguishing between
    // types with the same width. For example,
    // "long" and "long long" are both 64 bits on
    // 64-bit platforms. This may cause issues such as incorrect redeclarations
    // of system libraries, for example.
    //
    // To resolve this problem,
    // we rely on the "#cpp_type" field which contains a string
    // representation of the corresponding type in the original
    // program. If "#cpp_type" is not empty, we try to use
    // this information first, and only resort to inferring the type name
    // from its width otherwise.
    std::string cpp_type = src.get("#cpp_type").as_string();
    if (!cpp_type.empty() && width % 8 == 0)
    {
      std::replace(cpp_type.begin(), cpp_type.end(), '_', ' ');
      size_t sign_pos = cpp_type.find(" ");
      std::string sign_str = cpp_type.substr(0, sign_pos + 1);
      if (sign_str == "signed ")
        sign_str = "";
      std::string type_name = cpp_type.substr(sign_pos + 1, cpp_type.length());
      return q + sign_str + type_name + d;
    }

    // Since the above is unsuccessful at this point,
    // we try to infer the type "name" from its size.
    // Identifying the signedness first
    bool is_signed = src.id() == "signedbv";

    // We do not add the "signed" keyword for signed types.
    // Also types with the width of 1 bit or less cannot be "signed".
    // And if none of the above is the case we add the "unsigned" keyword.
    std::string sign_str = (is_signed && width > 1) ? "" : "unsigned ";

    if (width == config.ansi_c.int_width)
      return q + sign_str + "int" + d;

    if (width == config.ansi_c.long_int_width)
      return q + sign_str + "long int" + d;

    if (width == config.ansi_c.long_long_int_width)
      return q + sign_str + "long long int" + d;

    if (width == config.ansi_c.char_width)
      return q + sign_str + "char" + d;

    if (width == config.ansi_c.short_int_width)
      return q + sign_str + "short int" + d;

    // By this point we could not match the type width to any of the above.
    // So we treat it as a bit-vector of "custom" size.
    // We use now deprecated Clang extension "_ExtInt" (for backward
    // compatibility with ESBMC frontend that uses Clang11)
    // which has since been replaced by "_BigInt".
    //
    // Note that this can also be a source of issues for translating
    // struct/union's with bitfields for compiling with GCC that does
    // not have the "_ExtInt/_BigInt" syntactic constructs.
    return q + sign_str + "_ExtInt(" + std::to_string(width.to_uint64()) + ")" +
           d;
  }
  else if (src.id() == "floatbv" || src.id() == "fixedbv")
  {
    BigInt width = string2integer(src.width().as_string());

    if (width == config.ansi_c.single_width)
      return q + "float" + d;

    if (width == config.ansi_c.double_width)
      return q + "double" + d;

    if (width == config.ansi_c.long_double_width)
      return q + "long double" + d;

    assert(!"Unsupported width for floating point types");
  }
  else if (src.id() == "struct" || src.id() == "union")
  {
    const struct_union_typet &struct_union_type = to_struct_union_type(src);
    std::string dest = q;
    std::string tag = struct_union_type.tag().as_string();

    dest += tag;

    // Finally, adding the declarator (i.e., the declared symbol)
    return dest + " " + declarator;
  }
  else if (src.id() == "c_enum" || src.id() == "incomplete_c_enum")
  {
    std::string result = q + "enum";
    if (src.name() != "")
      result += " " + src.tag().as_string();
    result += d;
    return result;
  }
  else if (src.id() == "incomplete_struct")
  {
    std::string dest = q + "struct";
    const std::string &tag = src.tag().as_string();
    if (tag != "")
      dest += " " + tag;
    dest += d;
    return dest;
  }
  else if (src.id() == "pointer")
  {
    // This is a function pointer declaration.
    if (src.subtype().is_code())
    {
      // Can cast "src.subtype()" to "code_typet"
      const code_typet type = to_code_type(src.subtype());
      // Building the resulting string
      std::string dest = "";
      // Starting with the function parameters
      dest += "(";
      // Converting the function parameters just as
      // regular variable declarations (perhaps we should implement
      // a separate method in the future for this)
      for (unsigned int i = 0; i < type.arguments().size(); i++)
      {
        code_typet::argumentt arg = type.arguments()[i];
        std::string arg_name =
          get_name_shorthand(arg.get_identifier().as_string());

        dest += convert_rec(arg.type(), c_qualifierst(), arg_name);

        if (i != type.arguments().size() - 1)
          dest += ", ";
      }
      // Before finishing with the parameters list,
      // check if it is a variadic function
      if (type.has_ellipsis() && type.arguments().size() > 0)
        dest += ", ... ";
      dest += ")";

      // Now converting everything else (the return type and the function name)
      // "d" holds the declarator (e.g., function name)
      // "q" holds the type qualifier
      // "rst_q" holds the "__restrict" keyword if used
      // (unused so far but should be used)
      dest = q + convert_rec(
                   type.return_type(), c_qualifierst(), "(*" + d + ")" + dest);

      return dest;
    }
    else
    {
      // The base type might contain a struct/union tag of symbol type
      // instead of the actual underlying struct/union type.
      // Thus, we derive the underlying struct/union type through a
      // call to "base_type"
      typet subtype = src.subtype();
      base_type(subtype, ns);

      // Now can just convert
      // "d" holds the declarator (e.g., pointer name)
      // "q" holds the type qualifier
      // "rst_q" holds the "__restrict" keyword if used
      std::string dest =
        q + convert_rec(subtype, new_qualifiers, "*" + rst_q + d);
      return dest;
    }
  }
  else if (src.is_array())
  {
    std::string size_string =
      convert(static_cast<const exprt &>(src.size_irep()));

    std::string new_decl = d + "[" + size_string + "]";

    return convert_rec(src.subtype(), new_qualifiers, new_decl);
  }
  else if (src.id() == "incomplete_array")
  {
    return convert(src.subtype()) + " []";
  }
  else if (src.id() == "symbol")
  {
    const typet &followed = ns.follow(src);
    if (followed.id() == "struct" || followed.id() == "union")
    {
      std::string dest = q;
      const std::string &tag = followed.tag().as_string();
      if (tag != "")
        dest += " " + tag;
      dest += d;
      return dest;
    }
    return convert_rec(ns.follow(src), new_qualifiers, declarator);
  }
  else if (src.is_code())
  {
    // This is a function declaration.
    // Can cast "src" to "code_typet"
    const code_typet type = to_code_type(src);
    // Building the resulting string
    std::string dest = "";
    // Starting with the function parameters
    dest += "(";
    // Converting the function parameters just as
    // regular variable declarations (perhaps we should implement
    // a separate method in the future for this)
    for (unsigned int i = 0; i < type.arguments().size(); i++)
    {
      code_typet::argumentt arg = type.arguments()[i];
      std::string arg_name =
        get_name_shorthand(arg.get_identifier().as_string());

      dest += convert_rec(arg.type(), c_qualifierst(), arg_name);

      if (i != type.arguments().size() - 1)
        dest += ", ";
    }
    // Before finishing with the parameters list,
    // check if it is a variadic function
    if (type.has_ellipsis() && type.arguments().size() > 0)
      dest += ", ... ";
    dest += ")";

    // Now converting everything else (the return type and the function name)
    // "d" holds the declarator (e.g., function name)
    // "q" holds the type qualifier
    // "rst_q" holds the "__restrict" keyword if used
    // (unused so far but should be used)
    dest = q + convert_rec(type.return_type(), c_qualifierst(), d + dest);

    return dest;
  }

  unsigned precedence;
  return convert_norep((exprt &)src, precedence);
}

std::string expr2ct::convert_code_printf(const codet &src, unsigned indent)
{
  std::string dest = indent_str(indent) + "printf(";

  forall_operands (it, src)
  {
    unsigned p;
    std::string arg_str = convert(*it, p);

    if (it != src.operands().begin())
      dest += ", ";
    // TODO: [add] brackets, if necessary, depending on p
    dest += arg_str;
  }

  dest += ")";

  return dest;
}

std::string expr2ct::convert_code_free(const codet &src, unsigned indent)
{
  if (src.operands().size() != 1)
  {
    unsigned precedence;
    return convert_norep(src, precedence);
  }

  return indent_str(indent) + "free(" + convert(src.op0()) + ")";
}

std::string expr2ct::convert_code_return(const codet &src, unsigned indent)
{
  if (src.operands().size() != 0 && src.operands().size() != 1)
  {
    unsigned precedence;
    return convert_norep(src, precedence);
  }

  std::string dest = indent_str(indent);
  dest += "return";

  if (to_code_return(src).has_return_value())
    dest += " " + convert(src.op0());

  return dest;
}

std::string expr2ct::convert_code_assign(const codet &src, unsigned indent)
{
  unsigned int precedence = 15;
  std::string dest = convert(src.op0(), precedence);
  dest += "=";

  exprt rhs = src.op1();
  // Form a compound literal if assigning to a struct/union
  if (
    src.op1().id() == "array" || src.op1().id() == "array_of" ||
    src.op1().id() == "struct" || src.op1().id() == "union")
    rhs.make_typecast(src.op0().type());

  dest += convert(rhs, precedence);

  return indent_str(indent) + dest;
}

std::string expr2ct::convert_code_assert(const codet &src, unsigned indent)
{
  if (src.operands().size() != 1)
  {
    unsigned precedence;
    return convert_norep(src, precedence);
  }

  return indent_str(indent) + "assert(" + convert(src.op0()) + ")";
}

std::string expr2ct::convert_code_assume(const codet &src, unsigned indent)
{
  if (src.operands().size() != 1)
  {
    unsigned precedence;
    return convert_norep(src, precedence);
  }

  return indent_str(indent) + "assume(" + convert(src.op0()) + ")";
}

std::string expr2ct::convert_symbol(const exprt &src, unsigned &)
{
  const irep_idt &id = src.identifier();
  std::string dest;

  if (ns_collision.find(id) == ns_collision.end())
    dest = id_shorthand(src);
  else
    dest = id2string(id);

  if (src.id() == "next_symbol")
    dest = "next_symbol(" + dest + ")";

  // Replacing some characters in the names of identifiers with underscores
  std::replace(dest.begin(), dest.end(), '$', '_');
  std::replace(dest.begin(), dest.end(), '?', '_');
  std::replace(dest.begin(), dest.end(), '!', '_');
  std::replace(dest.begin(), dest.end(), '&', '_');
  std::replace(dest.begin(), dest.end(), '#', '_');
  std::replace(dest.begin(), dest.end(), '@', '_');
  std::replace(dest.begin(), dest.end(), '.', '_');
  std::replace(dest.begin(), dest.end(), ':', '_');
  std::replace(dest.begin(), dest.end(), '-', '_');

  return dest;
}

std::string expr2ct::convert(const exprt &src, unsigned &precedence)
{
  precedence = 16;

  if (src.id() == "+")
    return convert_binary(src, "+", precedence = 12, false);

  if (src.id() == "-")
  {
    if (src.operands().size() == 1)
      return convert_norep(src, precedence);

    return convert_binary(src, "-", precedence = 12, true);
  }

  else if (src.id() == "unary-")
  {
    if (src.operands().size() != 1)
      return convert_norep(src, precedence);

    return convert_unary(src, "-", precedence = 15);
  }

  else if (src.id() == "unary+")
  {
    if (src.operands().size() != 1)
      return convert_norep(src, precedence);

    return convert_unary(src, "+", precedence = 15);
  }

  else if (src.id() == "invalid-pointer")
  {
    return convert_function(src, "INVALID_POINTER", precedence = 15);
  }

  else if (src.id() == "invalid-object")
  {
    return "invalid_object";
  }

  else if (src.id() == "NULL-object")
  {
    return "0";
  }

  else if (src.id() == "infinity")
  {
    return convert_infinity(src, precedence = 15);
  }

  else if (src.id() == "builtin-function")
  {
    return src.identifier().as_string();
  }

  else if (src.id() == "pointer_object")
  {
    return convert_function(src, "__ESBMC_POINTER_OBJECT", precedence = 15);
  }

  else if (src.id() == "object_value")
  {
    return convert_function(src, "OBJECT_VALUE", precedence = 15);
  }

  else if (src.id() == "pointer_object_has_type")
  {
    return convert_pointer_object_has_type(src, precedence = 15);
  }

  else if (src.id() == "array_of")
  {
    return convert_array_of(src, precedence = 15);
  }

  else if (src.id() == "pointer_offset")
  {
    return convert_pointer_offset(src, precedence = 15);
  }

  else if (src.id() == "pointer_base")
  {
    return convert_function(src, "POINTER_BASE", precedence = 15);
  }

  else if (src.id() == "pointer_cons")
  {
    return convert_function(src, "POINTER_CONS", precedence = 15);
  }

  else if (src.id() == "same-object")
  {
    return convert_same_object(src, precedence = 15);
  }

  else if (src.id() == "valid_object")
  {
    return convert_function(src, "valid_object", precedence = 15);
  }

  else if (src.id() == "deallocated_object" || src.id() == "memory-leak")
  {
    return convert_function(src, "deallocated_object", precedence = 15);
  }

  else if (src.id() == "dynamic_object")
  {
    return convert_function(src, "dynamic_object", precedence = 15);
  }

  else if (src.id() == "is_dynamic_object")
  {
    return convert_function(src, "is_dynamic_object", precedence = 15);
  }

  else if (src.id() == "dynamic_size")
  {
    return convert_dynamic_size(src, precedence = 15);
  }

  else if (src.id() == "dynamic_type")
  {
    return convert_function(src, "dynamic_type", precedence = 15);
  }

  else if (src.id() == "isnan")
  {
    return convert_function(src, "isnan", precedence = 15);
  }

  else if (src.id() == "isfinite")
  {
    return convert_function(src, "isfinite", precedence = 15);
  }

  else if (src.id() == "isinf")
  {
    return convert_function(src, "isinf", precedence = 15);
  }

  else if (src.id() == "isnormal")
  {
    return convert_function(src, "isnormal", precedence = 15);
  }

  else if (src.id() == "signbit")
  {
    return convert_function(src, "signbit", precedence = 15);
  }

  else if (src.id() == "nearbyint")
  {
    return convert_function(src, "nearbyint", precedence = 15);
  }

  else if (src.id() == "popcount")
  {
    return convert_function(src, "popcount", precedence = 15);
  }

  else if (src.id() == "bswap")
  {
    return convert_function(src, "bswap", precedence = 15);
  }

  else if (src.id() == "builtin_va_arg")
  {
    return convert_function(src, "builtin_va_arg", precedence = 15);
  }

  else if (has_prefix(src.id_string(), "byte_extract"))
  {
    return convert_byte_extract(src, precedence = 15);
  }

  else if (has_prefix(src.id_string(), "byte_update"))
  {
    return convert_byte_update(src, precedence = 15);
  }

  else if (src.is_address_of())
  {
    if (src.operands().size() != 1)
      return convert_norep(src, precedence);
    if (src.op0().id() == "label")
      return "&&" + src.op0().get_string("identifier");
    else
      return convert_unary(src, "&", precedence = 15);
  }

  else if (src.id() == "dereference")
  {
    if (src.operands().size() != 1)
      return convert_norep(src, precedence);

    // Dereferencing a function pointer here
    if (src.type().is_code())
      return "(*" + convert_unary(src, "", precedence = 15) + ")";

    return convert_unary(src, "*", precedence = 15);
  }

  else if (src.id() == "index")
    return convert_index(src, precedence = 16);

  else if (src.id() == "member")
    return convert_member(src, precedence = 16);

  else if (src.id() == "array-member-value")
    return convert_array_member_value(src, precedence = 16);

  else if (src.id() == "struct-member-value")
    return convert_struct_member_value(src, precedence = 16);

  else if (src.id() == "sideeffect")
  {
    const irep_idt &statement = src.statement();
    if (statement == "preincrement")
      return convert_unary(src, "++", precedence = 15);
    if (statement == "predecrement")
      return convert_unary(src, "--", precedence = 15);
    else if (statement == "postincrement")
      return convert_unary_post(src, "++", precedence = 16);
    else if (statement == "postdecrement")
      return convert_unary_post(src, "--", precedence = 16);
    else if (statement == "assign+")
      return convert_binary(src, "+=", precedence = 2, true);
    else if (statement == "assign-")
      return convert_binary(src, "-=", precedence = 2, true);
    else if (statement == "assign*")
      return convert_binary(src, "*=", precedence = 2, true);
    else if (statement == "assign_div")
      return convert_binary(src, "/=", precedence = 2, true);
    else if (statement == "assign_mod")
      return convert_binary(src, "%=", precedence = 2, true);
    else if (statement == "assign_shl")
      return convert_binary(src, "<<=", precedence = 2, true);
    else if (statement == "assign_ashr")
      return convert_binary(src, ">>=", precedence = 2, true);
    else if (statement == "assign_bitand")
      return convert_binary(src, "&=", precedence = 2, true);
    else if (statement == "assign_bitxor")
      return convert_binary(src, "^=", precedence = 2, true);
    else if (statement == "assign_bitor")
      return convert_binary(src, "|=", precedence = 2, true);
    else if (statement == "assign")
      return convert_binary(src, "=", precedence = 2, true);
    else if (statement == "function_call")
      return convert_function_call(src, precedence);
    else if (statement == "malloc")
      return convert_malloc(src, precedence = 15);
    else if (statement == "realloc")
      return convert_realloc(src, precedence = 15);
    else if (statement == "alloca")
      return convert_alloca(src, precedence = 15);
    else if (statement == "printf")
      return convert_function(src, "printf", precedence = 15);
    else if (statement == "nondet")
      return convert_nondet(src, precedence = 15);
    else if (statement == "statement_expression")
      return convert_statement_expression(src, precedence = 15);
    else if (statement == "va_arg")
      return convert_function(src, "va_arg", precedence = 15);
    else
      return convert_norep(src, precedence);
  }

  else if (src.id() == "not")
    return convert_unary(src, "!", precedence = 15);

  else if (src.id() == "bitnot")
    return convert_unary(src, "~", precedence = 15);

  else if (src.id() == "*")
    return convert_binary(src, src.id_string(), precedence = 13, false);

  else if (src.id() == "/")
    return convert_binary(src, src.id_string(), precedence = 13, true);

  else if (src.id() == "mod")
    return convert_binary(src, "%", precedence = 13, true);

  else if (src.id() == "shl")
    return convert_binary(src, "<<", precedence = 11, true);

  else if (src.id() == "ashr" || src.id() == "lshr")
    return convert_binary(src, ">>", precedence = 11, true);

  else if (
    src.id() == "<" || src.id() == ">" || src.id() == "<=" || src.id() == ">=")
    return convert_binary(src, src.id_string(), precedence = 10, true);

  else if (src.id() == "notequal")
    return convert_binary(src, "!=", precedence = 9, true);

  else if (src.id() == "=")
    return convert_binary(src, "==", precedence = 9, true);

  else if (src.id() == "ieee_add")
    return convert_ieee_add(src, precedence = 12);

  else if (src.id() == "ieee_sub")
    return convert_ieee_sub(src, precedence = 12);

  else if (src.id() == "ieee_mul")
    return convert_ieee_mul(src, precedence = 13);

  else if (src.id() == "ieee_div")
    return convert_ieee_div(src, precedence = 13);

  else if (src.id() == "ieee_sqrt")
    return convert_ieee_sqrt(src, precedence = 15);

  else if (src.id() == "width")
    return convert_function(src, "width", precedence = 15);

  else if (src.id() == "byte_update_little_endian")
    return convert_function(src, "byte_update_little_endian", precedence = 15);

  else if (src.id() == "byte_update_big_endian")
    return convert_function(src, "byte_update_big_endian", precedence = 15);

  else if (src.id() == "abs")
    return convert_function(src, "abs", precedence = 15);

  else if (src.id() == "bitand")
    return convert_binary(src, "&", precedence = 8, false);

  else if (src.id() == "bitxor")
    return convert_binary(src, "^", precedence = 7, false);

  else if (src.id() == "bitor")
    return convert_binary(src, "|", precedence = 6, false);

  else if (src.is_and())
    return convert_binary(src, "&&", precedence = 5, false);

  else if (src.id() == "or")
    return convert_binary(src, "||", precedence = 4, false);

  else if (src.id() == "=>")
  {
    // Here we convert an implication A => B
    // in to an equivalent disjunction !A || B
    assert(src.operands().size() == 2);
    not_exprt not_expr(src.op0());
    or_exprt or_expr(not_expr, src.op1());
    return convert(or_expr, precedence);
  }

  else if (src.id() == "if")
    return convert_trinary(src, "?", ":", precedence = 3);

  else if (src.id() == "forall")
    return convert_quantifier(src, "forall", precedence = 2);

  else if (src.id() == "exists")
    return convert_quantifier(src, "exists", precedence = 2);

  else if (src.id() == "with")
    return convert_with(src, precedence = 2);

  else if (src.id() == "symbol")
    return convert_symbol(src, precedence);

  else if (src.id() == "next_symbol")
    return convert_symbol(src, precedence);

  else if (src.id() == "nondet_symbol")
    return convert_nondet_symbol(src, precedence);

  else if (src.id() == "predicate_symbol")
    return convert_predicate_symbol(src, precedence);

  else if (src.id() == "predicate_next_symbol")
    return convert_predicate_next_symbol(src, precedence);

  else if (src.id() == "quantified_symbol")
    return convert_quantified_symbol(src, precedence);

  else if (src.id() == "nondet_bool")
    return convert_nondet_bool(src, precedence);

  else if (src.id() == "object_descriptor")
    return convert_object_descriptor(src, precedence);

  else if (src.id() == "hoare")
    return convert_Hoare(src);

  else if (src.is_code())
    return convert_code(to_code(src));

  else if (src.id() == "constant")
    return convert_constant(src, precedence);

  else if (src.id() == "string-constant")
    return convert_constant(src, precedence);

  else if (src.id() == "struct")
    return convert_struct(src, precedence);

  else if (src.id() == "union")
    return convert_union(src, precedence);

  else if (src.is_array())
    return convert_array(src, precedence);

  else if (src.id() == "array-list")
    return convert_array_list(src, precedence);

  else if (src.id() == "typecast")
    return convert_typecast(src, precedence);

  else if (src.id() == "bitcast")
    return convert_bitcast(src, precedence);

  else if (src.id() == "implicit_address_of")
    return convert_implicit_address_of(src, precedence);

  else if (src.id() == "implicit_dereference")
    return convert_function(src, "implicit_dereference", precedence = 15);

  else if (src.id() == "comma")
    return convert_binary(src, ", ", precedence = 1, false);

  else if (src.id() == "cond")
    return convert_cond(src, precedence);

  else if (std::string(src.id_string(), 0, 9) == "overflow-")
    return convert_overflow(src, precedence);

  else if (src.id() == "unknown")
    return "*";

  else if (src.id() == "invalid")
    return "#";

  else if (src.id() == "extractbit")
    return convert_extractbit(src, precedence);

  else if (src.id() == "sizeof")
    return convert_sizeof(src, precedence);

  else if (src.id() == "concat")
    return convert_function(src, "concat", precedence = 15);

  else if (src.id() == "extract")
    return convert_extract(src);

  // no c language expression for internal representation
  return convert_norep(src, precedence);
}

std::string expr2ct::convert_typecast(const exprt &src, unsigned &precedence)
{
  precedence = 14;

  if (src.id() == "typecast" && src.operands().size() != 1)
    return convert_norep(src, precedence);

  // some special cases
  const typet &type = ns.follow(src.type());

  if (
    type.id() == "pointer" &&
    ns.follow(type.subtype()).id() == "empty" && // to (void *)?
    src.op0().is_zero())
    return "0";

  std::string dest;
  dest = "(" + convert(type) + ")";

  // Another special case here. ESBMC produces expressions
  // like "&{}[0]" and "&{1,2}[0]" which are illegal in C.
  // So we just replace it with the underlying constant array
  if (
    src.op0().is_address_of() && src.op0().op0().id() == "index" &&
    src.op0().op0().op0().id() == "constant")
    return dest + convert(src.op0().op0().op0());

  std::string tmp = convert(src.op0(), precedence);

  // better fix precedence
  if (
    src.op0().id() == "member" || src.op0().id() == "constant" ||
    src.op0().id() == "symbol" || src.op0().id() == "array_of" ||
    src.op0().id() == "struct" || src.op0().id() == "union")
    dest += tmp;
  else
    dest += "(" + tmp + ")";

  return dest;
}

std::string expr2ct::convert_code_decl(const codet &src, unsigned indent)
{
  if (src.operands().size() != 1 && src.operands().size() != 2)
  {
    unsigned precedence;
    return convert_norep(src, precedence);
  }

  std::string declarator = convert(src.op0());

  std::string dest = indent_str(indent);

  const symbolt *symbol = ns.lookup(to_symbol_expr(src.op0()).get_identifier());
  if (symbol)
  {
    if (
      symbol->file_local &&
      (src.op0().type().is_code() || symbol->static_lifetime))
      dest += "static ";
    else if (symbol->is_extern)
      dest += "extern ";

    if (symbol->type.is_code() && to_code_type(symbol->type).get_inlined())
      dest += "inline ";
  }

  dest += convert_rec(src.op0().type(), c_qualifierst(), declarator);

  // Checking if there is a non-empty initializer
  if (src.operands().size() == 2 && src.op1() != exprt())
    dest += "=" + convert(src.op1());

  return dest;
}

std::string expr2ct::convert_struct(const exprt &src, unsigned &precedence)
{
  const typet full_type = ns.follow(src.type());

  if (full_type.id() != "struct")
    return convert_norep(src, precedence);

  const struct_typet &struct_type = to_struct_type(full_type);
  const struct_union_typet::componentst &components = struct_type.components();

  if (components.size() != src.operands().size())
    return convert_norep(src, precedence);

  return convert_struct_union_body(src, src.operands(), components);
}

std::string expr2ct::convert_union(const exprt &src, unsigned &precedence)
{
  const typet full_type = ns.follow(src.type());

  if (full_type.id() != "union")
    return convert_norep(src, precedence);

  const exprt::operandst &operands = src.operands();
  const irep_idt &init = src.component_name();

  if (operands.size() == 1)
  {
    // Fedor: The following assert is triggered if the initialized
    // member of the union is an anonymous struct/union, which
    // is perfectly legal. So we intercept this scenario and go
    // straight into converting the initializer.
    if (init.empty())
      return convert(src.op0());

    // Initializer known
    assert(!init.empty());

    std::string dest = "(" + to_union_type(full_type).tag().as_string() + ")";
    dest += "{";
    std::string tmp = convert(src.op0());
    dest += ".";
    dest += init.as_string();
    dest += "=";
    dest += tmp;
    dest += "}";

    return dest;
  }
  else
  {
    // Initializer unknown, expect operands assigned to each member and convert
    // all of them
    assert(init.empty());
    return convert_struct_union_body(
      src, operands, to_union_type(full_type).components());
  }
}

// This produces a compound literal from the given
// struct/union body
std::string expr2ct::convert_struct_union_body(
  const exprt &src,
  const exprt::operandst &operands,
  const struct_union_typet::componentst &components)
{
  size_t n = components.size();
  assert(n == operands.size());
  assert(src.type().id() == "struct" || src.type().id() == "union");

  std::string dest = "";
  struct_union_typet struct_union_type;

  if (src.type().id() == "struct")
    struct_union_type = to_struct_type(src.type());

  if (src.type().id() == "union")
    struct_union_type = to_union_type(src.type());

  std::string tag = struct_union_type.tag().as_string();

  dest += "(" + tag + ")";

  //if(struct_union_type.components().size() == 0)
  //  return dest;

  dest += "{";

  for (size_t i = 0; i < n; i++)
  {
    const auto &operand = operands[i];
    const auto &component = components[i];

    if (component.type().is_code())
      continue;

    // Fedor: this seems to be never working. Perhaps
    // the information gets lost after migrating from irep to
    // irep2 and back. Hence, we check this in a different way.
    // See below.
    if (component.get_is_padding())
      continue;

    // Do not output the padding members
    if (is_padding(component.get_name().as_string()))
      continue;

    std::string tmp = convert(operand);

    dest += ".";
    dest += component.get_name().as_string();
    dest += "=";
    dest += tmp;

    if (i < n - 1)
      dest += ", ";
  }

  dest += "}";

  return dest;
}

std::string expr2ct::convert_member(const exprt &src, unsigned precedence)
{
  if (src.operands().size() != 1)
    return convert_norep(src, precedence);

  unsigned p;
  std::string dest;

  if (src.op0().id() == "dereference" && src.operands().size() == 1)
  {
    std::string op = convert(src.op0().op0(), p);

    dest += "(" + op + ")";
    dest += "->";
  }
  else
  {
    std::string op = convert(src.op0(), p);

    if (precedence > p)
      dest += '(';
    dest += op;
    if (precedence > p)
      dest += ')';

    dest += '.';
  }

  const typet &full_type = ns.follow(src.op0().type());

  // It might be a flattened union
  // This will look very odd when printing, but it's better then
  // the norep output
  if (full_type.id() == "array")
    return convert_array(src, precedence);

  if (full_type.id() != "struct" && full_type.id() != "union")
    return convert_norep(src, precedence);

  const struct_typet &struct_type = to_struct_type(full_type);

  const exprt comp_expr = struct_type.get_component(src.component_name());

  if (comp_expr.is_nil())
    return convert_norep(src, precedence);

  dest += comp_expr.name().as_string();

  return dest;
}

std::string expr2ct::convert_constant(const exprt &src, unsigned &precedence)
{
  const typet &type = ns.follow(src.type());
  const std::string &cformat = src.cformat().as_string();
  const std::string &value = src.value().as_string();
  std::string dest;

  if (cformat != "")
    dest = cformat;
  else if (src.id() == "string-constant")
  {
    dest = '"';
    MetaString(dest, value);
    dest += '"';
  }
  else if (type.id() == "c_enum" || type.id() == "incomplete_c_enum")
  {
    BigInt int_value = string2integer(value);
    BigInt i = 0;
    const irept &body = type.body();

    forall_irep (it, body.get_sub())
    {
      if (i == int_value)
      {
        dest = it->name().as_string();
        return dest;
      }

      ++i;
    }

    // failed...
    dest = "enum(" + value + ")";

    return dest;
  }
  else if (type.id() == "bv")
    dest = value;
  else if (type.is_bool())
  {
    dest = src.is_true() ? "1" : "0";
  }
  else if (type.id() == "unsignedbv" || type.id() == "signedbv")
  {
    BigInt int_value = binary2integer(value, type.id() == "signedbv");
    dest = integer2string(int_value);
  }
  else if (type.id() == "floatbv")
  {
    dest = ieee_floatt(to_constant_expr(src)).to_ansi_c_string();

    if (dest != "" && isdigit(dest[dest.size() - 1]))
    {
      if (src.type() == float_type())
        dest += "f";
      else if (src.type() == long_double_type())
        dest += "l";
    }
  }
  else if (type.id() == "fixedbv")
  {
    dest = fixedbvt(to_constant_expr(src)).to_ansi_c_string();

    if (dest != "" && isdigit(dest[dest.size() - 1]))
    {
      if (src.type() == float_type())
        dest += "f";
      else if (src.type() == long_double_type())
        dest += "l";
    }
  }
  else if (is_array_like(type))
  {
    dest = "{";

    forall_operands (it, src)
    {
      std::string tmp = convert(*it);

      if ((it + 1) != src.operands().end())
        tmp += ", ";

      dest += tmp;
    }

    dest += "}";
  }
  else if (type.id() == "pointer")
  {
    if (value == "NULL")
      dest = "((" + convert(type) + ")0)";
    else if (value == "INVALID" || std::string(value, 0, 8) == "INVALID-")
      dest = value;
    else
      return convert_norep(src, precedence);
  }
  else
    return convert_norep(src, precedence);

  return dest;
}

// This is an attempt of encoding and translating ESBMC intrinsic
// function "_Bool __ESBMC_same_object(void *, void *)" that returns true
// when both pointers point to the same object in memory.
std::string expr2ct::convert_same_object(const exprt &src, unsigned &precedence)
{
  assert(src.operands().size() == 2);
  expr2tc new_src;
  migrate_expr(src, new_src);

  assert(is_same_object2t(new_src));
  const same_object2t &same = to_same_object2t(new_src);

  assert(
    is_pointer_type(same.side_1->type) && is_pointer_type(same.side_2->type));

  if (is_address_of2t(same.side_2))
  {
    const address_of2t &addr = to_address_of2t(same.side_2);
    if (is_array_type(addr.ptr_obj->type))
    {
      const array_type2t &arr_type = to_array_type(addr.ptr_obj->type);
      // this arr_size is equal to the number of elements of the given array
      expr2tc arr_size = arr_type.array_size;
      expr2tc gt = greaterthan2tc(addr.ptr_obj, same.side_1);
      expr2tc gt2 = greaterthan2tc(
        add2tc(same.side_1->type, same.side_1, gen_ulong(1)),
        add2tc(arr_type.subtype, addr.ptr_obj, arr_size));
      expr2tc in_bounds = or2tc(gt, gt2);
      simplify(in_bounds);
      return convert(migrate_expr_back(in_bounds), precedence);
    }
  }

  expr2tc eq = equality2tc(same.side_1, same.side_2);
  return convert(migrate_expr_back(not2tc(eq)), precedence);
}

std::string expr2ct::convert_pointer_offset(
  const exprt &src,
  unsigned &precedence [[maybe_unused]])
{
  std::string dest = "pointer_offset";
  dest += '(';

  forall_operands (it, src)
  {
    unsigned p;
    std::string op = convert(*it, p);

    if (it != src.operands().begin())
      dest += ", ";

    dest += op;
  }

  dest += ')';

  return dest;
}

std::string expr2ct::convert_malloc(const exprt &src, unsigned &precedence)
{
  if (src.operands().size() != 1)
    return convert_norep(src, precedence);

  unsigned p0;
  std::string op0 = convert(src.op0(), p0);

  std::string dest = "malloc";
  dest += '(';
  dest += op0;
  dest += ')';

  return dest;
}

std::string expr2ct::convert_realloc(const exprt &src, unsigned &precedence)
{
  if (src.operands().size() != 1)
    return convert_norep(src, precedence);

  unsigned p0, p1;
  std::string op0 = convert(src.op0(), p0);
  std::string size = convert((const exprt &)src.cmt_size(), p1);

  std::string dest = "realloc";
  dest += '(';
  dest += op0;
  dest += ", ";
  dest += size;
  dest += ')';

  return dest;
}

std::string expr2ct::convert_alloca(const exprt &src, unsigned &precedence)
{
  if (src.operands().size() != 1)
    return convert_norep(src, precedence);

  unsigned p0;
  std::string op0 = convert(src.op0(), p0);

  std::string dest = "alloca";
  dest += '(';
  dest += op0;
  dest += ')';

  return dest;
}

// This method contains all "__VERIFIER_nondet_" cases
// that appear in "esbmc_intrinsics.h".
std::string expr2ct::convert_nondet(const exprt &src, unsigned &precedence)
{
  if (src.operands().size() != 0)
    return convert_norep(src, precedence);

  std::string type_str = "";
  if (src.type().is_bool())
  {
    type_str += "bool";
  }
  else if (src.type().id() == "signedbv" || src.type().id() == "unsignedbv")
  {
    BigInt width = string2integer(src.type().width().as_string());

    if (src.type().id() == "unsignedbv")
      type_str += "u";

    if (width == config.ansi_c.int_width)
      type_str += "int";

    if (width == config.ansi_c.long_int_width)
      type_str += "long";

    if (width == config.ansi_c.char_width)
      type_str += "char";

    if (width == config.ansi_c.short_int_width)
      type_str += "short";
  }
  else if (src.type().id() == "floatbv" || src.type().id() == "fixedbv")
  {
    BigInt width = string2integer(src.type().width().as_string());

    if (width == config.ansi_c.single_width)
      type_str += "float";
    if (width == config.ansi_c.double_width)
      type_str += "double";
  }

  std::string dest = "__VERIFIER_nondet_" + type_str + "()";
  return dest;
}

std::string expr2ct::convert_array_of(const exprt &src, unsigned precedence)
{
  if (src.operands().size() != 1)
    return convert_norep(src, precedence);

  std::string dest = "";
  dest += "{";
  if (!src.op0().is_zero() && !src.op0().is_false())
    dest += convert(src.op0());
  dest += "}";

  return dest;
}

std::string
expr2ct::convert_dynamic_size(const exprt &src, unsigned &precedence)
{
  if (src.operands().size() != 1)
    return convert_norep(src, precedence);

  std::string dest = "dynamic_size";
  dest += "(";
  if (!src.op0().is_zero() && !src.op0().is_false())
  {
    dest += convert(src.op0());
  }
  dest += ")";

  return dest;
}

std::string expr2ct::convert_infinity(
  const exprt &src [[maybe_unused]],
  unsigned &precedence [[maybe_unused]])
{
  // Fedor: just an arbitrary value for now
  return "__ESBMC_INF_SIZE";
}

std::string expr2ct::convert_ieee_add(const exprt &src, unsigned &precedence)
{
  if (src.operands().size() != 2)
    return convert_norep(src, precedence);

  return "(" + convert(src.op0()) + ")+(" + convert(src.op1()) + ")";
}

std::string expr2ct::convert_ieee_sub(const exprt &src, unsigned &precedence)
{
  if (src.operands().size() != 2)
    return convert_norep(src, precedence);

  return "(" + convert(src.op0()) + ")-(" + convert(src.op1()) + ")";
}

std::string expr2ct::convert_ieee_mul(const exprt &src, unsigned &precedence)
{
  if (src.operands().size() != 2)
    return convert_norep(src, precedence);

  return "(" + convert(src.op0()) + ")*(" + convert(src.op1()) + ")";
}

std::string expr2ct::convert_ieee_div(const exprt &src, unsigned &precedence)
{
  if (src.operands().size() != 2)
    return convert_norep(src, precedence);

  return "(" + convert(src.op0()) + ")/(" + convert(src.op1()) + ")";
}

std::string expr2ct::convert_ieee_sqrt(const exprt &src, unsigned &precedence)
{
  if (src.operands().size() != 1)
    return convert_norep(src, precedence);

  return "sqrt(" + convert(src.op0()) + ")";
}
