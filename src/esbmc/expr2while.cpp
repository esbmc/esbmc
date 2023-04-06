#include <expr2while.h>
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

#include <clang-c-frontend/expr2c.h>

#include <exception>

class unsupported_error : public std::logic_error
{
public:
  unsupported_error(std::string what) : std::logic_error(what) {}
};

class norep_error : public std::logic_error
{
public:
  norep_error(std::string what) : std::logic_error(what) {}
};

int convert_to_while(
  contextt &context,
  std::string rapid_file_name){

  try{

    symbol_listt symbol_list;
    context.Foreach_operand_in_order([&symbol_list](symbolt &s) {
      if(!s.is_type && s.type.is_code())
        symbol_list.push_back(&s);
    });

    // we are only interested in translating the main 
    // function to While, not ESBMC intrinsics
    // While cannot (yet) support function calls
    auto get_func_name = [](std::string esbmc_name){
      size_t pos = esbmc_name.find_last_of("@");
      if (pos != std::string::npos){
        return esbmc_name.substr(pos + 1);
      }
      return esbmc_name;
    };

    bool main_found = false;

    symbolt* main;

    for(auto &it : symbol_list)
    {
      irep_idt identifier = it->id;  
      std::string identifier_str(identifier.c_str());
      if(get_func_name(identifier_str) == "main"){
        main = it;
        main_found = true;
        break;
      }
    }

    // TODO is below correct behaviour?
    if(!main_found){
      throw unsupported_error("could not convert to While as no main found");
    }

    const codet &code = to_code(main->value);

    expr2whilet expr2while(namespacet(context), rapid_file_name);
    expr2while.convert_main(code);
    return 0;
  
  } catch( const unsupported_error& e ) {
    log_error(e.what());
    return 6;
  } catch( const norep_error& e ) {
    log_error(e.what());
    return 6;
  } catch(...){
    log_error("unkown error occurred whilst converting to Rapid input format");
    return 6;
  }

}


std::string expr2whilet::id_shorthand(const exprt &expr) const
{
  const irep_idt &identifier = expr.identifier();
  const symbolt *symbol = ns.lookup(identifier);

  if(symbol)
    return id2string(symbol->name);

  std::string sh = id2string(identifier);

  std::string::size_type pos = sh.rfind("@");
  if(pos != std::string::npos)
    sh.erase(0, pos + 1);

  return sh;
}

void expr2whilet::get_symbols(const exprt &expr)
{
  if(expr.id() == "symbol")
    symbols.insert(expr);

  forall_operands(it, expr)
    get_symbols(*it);
}

/*void expr2whilet::get_shorthands(const exprt &expr)
{
  get_symbols(expr);

  for(const auto &symbol : symbols)
  {
    std::string sh = id_shorthand(symbol);

    std::pair<std::map<irep_idt, exprt>::iterator, bool> result =
      shorthands.insert(std::pair<irep_idt, exprt>(sh, symbol));

    if(!result.second)
      if(result.first->second != symbol)
      {
        ns_collision.insert(symbol.identifier());
        ns_collision.insert(result.first->second.identifier());
      }
  }
}*/

void expr2whilet::convert(const typet &src)
{
  convert_rec(src, c_qualifierst());
}

void expr2whilet::convert_rec(
  const typet &src,
  const c_qualifierst &qualifiers)
{

  c_qualifierst new_qualifiers(qualifiers);
  new_qualifiers.read(src);

  std::string q = new_qualifiers.as_string();

  if(src.is_bool())
  {
    throw unsupported_error("Rapid does not support Boolean type currently");
  }
  if(src.id() == "empty")
  {
    log_error("Rapid does not support void type currently");
  }
  else if(src.id() == "signedbv" || src.id() == "unsignedbv")
  {
    rapid_file << "Int";
  }
  else if(src.id() == "floatbv" || src.id() == "fixedbv")
  {
    rapid_file << "Int";
  }
  else if(src.id() == "struct")
  {
    //AYB TODO
    /*const struct_typet &struct_type = to_struct_type(src);

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
    res = dest;*/
  }
  else if(src.id() == "incomplete_struct")
  {
    // AYB what is this?
    /*std::string dest = q + "struct";
    const std::string &tag = src.tag().as_string();
    if(tag != "")
      dest += " " + tag;
    dest += d;
    res = dest;*/
  }
  else if(src.id() == "union")
  {
    throw unsupported_error("Rapid does not support union type");
  }
  else if(src.id() == "c_enum" || src.id() == "incomplete_c_enum")
  {
    throw unsupported_error("Rapid does not support enumeration types");
  }
  else if(src.id() == "pointer")
  {
    if(src.subtype().is_code())
    {
      throw unsupported_error("Rapid does not support function pointers");
    }

    convert(src.subtype());
    rapid_file << "*";
  }
  else if(src.is_array())
  {
    // Rapid only supports infinite size arrays
    // we just drop the size here
    // Means no reasoning about out of bounds is possible..
    convert(src.subtype());
    rapid_file << "[]";
  }
  /** int vector [3]
   *   /          |
   * type        size
   */
  else if(src.is_vector())
  {
    // AYB TODO treat same as array?
    /*std::string size_string =
      convert(static_cast<const exprt &>(src.size_irep()));
    res = convert(src.subtype()) + " vector [" + size_string + "]" + d;*/
  }
  else if(src.id() == "symbol")
  {
    // AYB what is this case?
    /*const typet &followed = ns.follow(src);
    if(followed.id() == "struct")
    {
      std::string dest = q;
      const std::string &tag = followed.tag().as_string();
      if(tag != "")
        dest += " " + tag;
      dest += d;
      res = dest;
    }

    if(followed.id() == "union")
    {
      std::string dest = q;
      const std::string &tag = followed.tag().as_string();
      if(tag != "")
        dest += " " + tag;
      dest += d;
      res = dest;
    }

    return convert_rec(ns.follow(src), new_qualifiers, declarator);*/
  }
  else if(src.is_code())
  {
    // What is this? Looks like sme sort of function type?
    /*const typet &return_type = (typet &)src.return_type();

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
    res = dest;*/
  } else {
    log_error("non-supported type");
  }
}

void expr2whilet::convert_typecast(const exprt &src, unsigned &precedence)
{
  precedence = 14;

  if(src.id() == "typecast" && src.operands().size() != 1)
    log_error("typecast has an incorrect number of operands. translation aborted");

  // just convert the thing inside and hope for the best for now!
  convert(src.op0(), precedence);

  // some special cases

  /*const typet &type = ns.follow(src.type());

  std::string res;

  if(
    type.id() == "pointer" &&
    ns.follow(type.subtype()).id() == "empty" && // to (void *)?
    src.op0().is_zero())
    res = "0";

  std::string dest;
  if(type.id() == "struct")
  {
    std::string dest;
    const std::string &tag = type.tag().as_string();
    assert(tag != "");
    dest += " " + tag;
    res = dest;
  }
  if(type.id() == "union")
  {
    std::string dest;
    const std::string &tag = type.tag().as_string();
    assert(tag != "");
    dest += " " + tag;
  }
  else
    dest = "(" + convert(type) + ")";

  std::string tmp = convert(src.op0(), precedence);

  if(
    src.op0().id() == "member" || src.op0().id() == "constant" ||
    src.op0().id() == "symbol") // better fix precedence
    dest += tmp;
  else
    dest += '(' + tmp + ')';

  res = dest;*/
}

/*
std::string
expr2whilet::convert_implicit_address_of(const exprt &src, unsigned &precedence)
{
  if(src.operands().size() != 1)
    return convert_norep(src, precedence);

  return convert(src.op0(), precedence);
}*/

void expr2whilet::convert_trinary(
  const exprt &src,
  const std::string &symbol1,
  const std::string &symbol2,
  unsigned precedence)
{
  throw unsupported_error("Currently the translation does not work with trinary operator used in expression: " + expr2c(src, ns));
}

void expr2whilet::convert_binary(
  const exprt &src,
  const std::string &symbol,
  unsigned precedence,
  bool full_parentheses)
{
  if(src.operands().size() < 2)
    log_error("binary operator must have two operands");


  // AYB TODO precedence?

  auto op0 = src.op0();
  auto op1 = src.op1();

  unsigned p;

  if(symbol == "+=" || symbol == "-=" || 
     symbol == "*=" || symbol == "%="){
 
    std::string op;
    if(symbol == "+=") op = " + ";
    if(symbol == "-=") op = " - ";
    if(symbol == "*=") op = " * ";
    if(symbol == "%=") op = " mod ";

    convert(op0, p);
    
    rapid_file << " = ";

    convert(op0, p);

    rapid_file << op;

    convert(op1, p);

  } else {

    convert(op0, p);

    rapid_file << " " << symbol << " ";

    convert(op1, p);
  
  }
}

void expr2whilet::convert_unary(
  const exprt &src,
  const std::string &symbol,
  unsigned precedence)
{
  if(src.operands().size() != 1)
    throw norep_error("incorrect number of operands for unary symbol " + symbol);

  unsigned p;
  if(symbol == "-"){
    // unary minus
    rapid_file << "(0 - ";
    convert(src.op0(), p);
    rapid_file << ")";
  } else if (symbol == "+"){
    convert(src.op0(), p);
  } else if (symbol == "*"){
    rapid_file << "*";
    convert(src.op0(), p);    
  } else if (symbol == "&"){
    // location of
    // TODO, Rapid only allows taking location of
    // on the right hand side of an assignment
    rapid_file << "#";
    convert(src.op0(), p);
  } else {
    throw unsupported_error("unsupported unary symbol " + symbol);
  }
}

void expr2whilet::convert_malloc(const exprt &src, unsigned &precedence)
{
  if(src.operands().size() != 1)
    throw norep_error("malloc has an incorrect number of operands. translation aborted");

  // TODO Rapid only support malloc on right of assignment
  rapid_file << "malloc()";
}

void expr2whilet::convert_nondet(const exprt &src, unsigned &precedence)
{
  if(src.operands().size() != 0)
    log_error("nondet has an incorrect number of operands. translation aborted");

  //return "NONDET(" + convert(src.type()) + ")";
}

/*std::string
expr2whilet::convert_statement_expression(const exprt &src, unsigned &precedence)
{
  if(
    src.operands().size() != 1 || to_code(src.op0()).get_statement() != "block")
    return convert_norep(src, precedence);

  return "(" + convert_code(to_code_block(to_code(src.op0())), 0) + ")";
}*/

void
expr2whilet::convert_function(const exprt &src, const std::string &name, unsigned)
{
  /*std::string dest = name;
  dest += '(';

  forall_operands(it, src)
  {
    unsigned p;
    std::string op = convert(*it, p);

    if(it != src.operands().begin())
      dest += ", ";

    dest += op;
  }

  dest += ')';*/

  //return dest;
}

/*void expr2whilet::convert_array_of(const exprt &src, unsigned precedence)
{
  if(src.operands().size() != 1)
    return convert_norep(src, precedence);

  return "ARRAY_OF(" + convert(src.op0()) + ')';
}
*/

void expr2whilet::convert_unary_post(
  const exprt &src,
  const std::string &symbol,
  unsigned precedence)
{
  if(src.operands().size() != 1)
    throw norep_error("incorrect number of operands for unary postfix operator " + symbol);

  unsigned p;
  convert(src.op0(), p);

  rapid_file << " = ";

  convert(src.op0(), p);

  if(symbol == "++"){
    rapid_file << " + ";
  } else if (symbol == "--"){
    rapid_file << " - ";
  } else {
    throw unsupported_error("not supported postfix symbol " + symbol);
  }

  rapid_file << "1";
}

void expr2whilet::convert_index(const exprt &src, unsigned precedence)
{
 /* if(src.operands().size() != 2)
    return convert_norep(src, precedence);

  unsigned p;
  std::string op = convert(src.op0(), p);

  std::string dest;
  if(precedence > p)
    dest += '(';
  dest += op;
  if(precedence > p)
    dest += ')';

  dest += '[';
  dest += convert(src.op1());
  dest += ']';

  return dest;*/
}

void expr2whilet::convert_member(const exprt &src, unsigned precedence)
{
  if(src.operands().size() != 1)
    log_error("member has an incorrect number of operands. translation aborted");

 /* unsigned p;
  std::string dest;

  if(src.op0().id() == "dereference" && src.operands().size() == 1)
  {
    std::string op = convert(src.op0().op0(), p);

    if(precedence > p)
      dest += '(';
    dest += op;
    if(precedence > p)
      dest += ')';

    dest += "->";
  }
  else
  {
    std::string op = convert(src.op0(), p);

    if(precedence > p)
      dest += '(';
    dest += op;
    if(precedence > p)
      dest += ')';

    dest += '.';
  }*/

  const typet &full_type = ns.follow(src.op0().type());

  // It might be an flattened union
  // This will look very odd when printing, but it's better then
  // the norep output
  /*if(full_type.id() == "array")
    return convert_array(src, precedence);

  if(full_type.id() != "struct" && full_type.id() != "union")
    return convert_norep(src, precedence);

  const struct_typet &struct_type = to_struct_type(full_type);

  const exprt comp_expr = struct_type.get_component(src.component_name());

  if(comp_expr.is_nil())
    return convert_norep(src, precedence);

  dest += comp_expr.pretty_name().as_string();*/

  //return dest;
}

/*std::string
expr2whilet::convert_array_member_value(const exprt &src, unsigned precedence)
{
  if(src.operands().size() != 1)
    return convert_norep(src, precedence);

  return "[]=" + convert(src.op0());
}

std::string
expr2whilet::convert_struct_member_value(const exprt &src, unsigned precedence)
{
  if(src.operands().size() != 1)
    return convert_norep(src, precedence);

  return "." + src.name().as_string() + "=" + convert(src.op0());
}

void expr2whilet::convert_norep(const exprt &src, unsigned &)
{
  return src.pretty(0);
}*/

void expr2whilet::convert_symbol(const exprt &src, unsigned &)
{
  const irep_idt &id = src.identifier();
  std::string dest;

  if(ns_collision.find(id) == ns_collision.end())
    dest = id_shorthand(src);
  else
    dest = id2string(id);

  rapid_file << dest;
}

/*void expr2whilet::convert_nondet_symbol(const exprt &src, unsigned &)
{
  const std::string &id = src.identifier().as_string();
  return "nondet_symbol(" + id + ")";
}

void expr2whilet::convert_predicate_symbol(const exprt &src, unsigned &)
{
  const std::string &id = src.identifier().as_string();
  return "ps(" + id + ")";
}

void expr2whilet::convert_predicate_next_symbol(const exprt &src, unsigned &)
{
  const std::string &id = src.identifier().as_string();
  return "pns(" + id + ")";
}

void expr2whilet::convert_quantified_symbol(const exprt &src, unsigned &)
{
  const std::string &id = src.identifier().as_string();
  return id;
}

void expr2whilet::convert_nondet_bool(const exprt &, unsigned &)
{
  return "nondet_bool()";
}*

std::string
expr2whilet::convert_object_descriptor(const exprt &src, unsigned &precedence)
{
  if(src.operands().size() != 2)
    return convert_norep(src, precedence);

  std::string result = "<";

  result += convert(src.op0());
  result += ", ";
  result += convert(src.op1());
  result += ", ";
  result += convert(src.type());

  result += ">";

  return result;
}*/

void expr2whilet::convert_constant(const exprt &src, unsigned &precedence)
{
  const typet &type = ns.follow(src.type());
  const std::string &cformat = src.cformat().as_string();
  const std::string &value = src.value().as_string();

  std::string src_id(src.id().c_str());

  if(cformat != ""){
    rapid_file << cformat;
  }
  else if(src.id() == "string-constant")
  {
    log_error("Rapid does not support string constants");
  }
  else if(type.id() == "c_enum" || type.id() == "incomplete_c_enum")
  {
    log_error("Rapid does not support enumeration types");
  }
  else if(type.id() == "bv")
  {
    rapid_file << value;
  }
  else if(type.is_bool())
  {
    log_error("Rapid does not support Boolean type");
  }
  else if(type.id() == "unsignedbv" || type.id() == "signedbv")
  {
    BigInt int_value = binary2integer(value, type.id() == "signedbv");
    rapid_file << integer2string(int_value);
  }
  else if(type.id() == "floatbv" || type.id() == "fixedbv")
  {
    log_error("Rapid does not support fixed and floating point types");
  }
  else if(is_array_like(type))
  {
  // AYB todo
   /* dest = "{ ";

    forall_operands(it, src)
    {
      std::string tmp = convert(*it);

      if((it + 1) != src.operands().end())
      {
        tmp += ", ";
        if(tmp.size() > 40)
          tmp += "\n    ";
      }

      dest += tmp;
    }

    dest += " }";*/
  }
  else if(type.id() == "pointer")
  {
    // AYB TODO
    /*if(value == "NULL")
      rapid_file <<
    else if(value == "INVALID" || std::string(value, 0, 8) == "INVALID-")
      dest = value;
    else
      return convert_norep(src, precedence);*/
  }
  else {
    log_error("Unsupported constant with id " + src_id);
  }
  //return dest;
}

void expr2whilet::convert_struct_union_body(
  const exprt::operandst &operands,
  const struct_union_typet::componentst &components)
{
/*  size_t n = components.size();
  assert(n == operands.size());

  std::string dest = "{ ";

  bool first = true;
  bool newline = false;
  unsigned last_size = 0;

  for(size_t i = 0; i < n; i++)
  {
    const auto &operand = operands[i];
    const auto &component = components[i];

    if(component.type().is_code())
      continue;

    if(component.get_is_padding())
      continue;

    if(first)
      first = false;
    else
    {
      dest += ",";

      if(newline)
        dest += "\n    ";
      else
        dest += " ";
    }

    std::string tmp = convert(operand);

    if(last_size + 40 < dest.size())
    {
      newline = true;
      last_size = dest.size();
    }
    else
      newline = false;

    dest += ".";
    dest += component.pretty_name().as_string();
    dest += "=";
    dest += tmp;
  }

  dest += " }";*/

  //return dest;
}

void expr2whilet::convert_struct(const exprt &src, unsigned &precedence)
{
 /* const typet full_type = ns.follow(src.type());

  if(full_type.id() != "struct")
    return convert_norep(src, precedence);

  const struct_typet &struct_type = to_struct_type(full_type);
  const struct_union_typet::componentst &components = struct_type.components();

 // if(components.size() != src.operands().size())
 //   return convert_norep(src, precedence);

  convert_struct_union_body(src.operands(), components);*/
}

/*void expr2whilet::convert_union(const exprt &src, unsigned &precedence)
{
  const typet full_type = ns.follow(src.type());

  if(full_type.id() != "union")
    return convert_norep(src, precedence);

  const exprt::operandst &operands = src.operands();
  const irep_idt &init [[maybe_unused]] = src.component_name();

  if(operands.size() == 1)
  {
    // Initializer known 
    assert(!init.empty());
    std::string dest = "{ ";

    std::string tmp = convert(src.op0());

    dest += ".";
    dest += init.as_string();
    dest += "=";
    dest += tmp;

    dest += " }";

    return dest;
  }
  else
  {
    // Initializer unknown, expect operands assigned to each member and convert
    // all of them 
    assert(init.empty());
    return convert_struct_union_body(
      operands, to_union_type(full_type).components());
  }
}

void expr2whilet::convert_array(const exprt &src, unsigned &)
{
  std::string dest = "{ ";

  forall_operands(it, src)
  {
    std::string tmp;

    if(it->is_not_nil())
      tmp = convert(*it);

    if((it + 1) != src.operands().end())
    {
      tmp += ", ";
      if(tmp.size() > 40)
        tmp += "\n    ";
    }

    dest += tmp;
  }

  dest += " }";

  return dest;
}

void expr2whilet::convert_array_list(const exprt &src, unsigned &precedence)
{
  std::string dest = "{ ";

  if((src.operands().size() % 2) != 0)
    return convert_norep(src, precedence);

  forall_operands(it, src)
  {
    std::string tmp1 = convert(*it);

    it++;

    std::string tmp2 = convert(*it);

    std::string tmp = "[" + tmp1 + "]=" + tmp2;

    if((it + 1) != src.operands().end())
    {
      tmp += ", ";
      if(tmp.size() > 40)
        tmp += "\n    ";
    }

    dest += tmp;
  }

  dest += " }";

  return dest;
}*/

void expr2whilet::convert_function_call(const exprt &src, unsigned &)
{
  if(src.operands().size() != 2)
  {
    log_error("incorrect number of operands for function call");
  }

  std::string function_str = expr2c(src.op0(), ns);
  
  unsigned p;

  if(function_str == "__ESBMC_assume"){
    rapid_file << "assume(";
    convert(src.op1().op0(), p);
    rapid_file << ")";
    return;
  }
  if(function_str == "__ESBMC_assert"){
    rapid_file << "assert(";
    convert(src.op1().op0(), p);
    rapid_file << ")";
    return;
  }  

  log_error("Rapid does not support function calls");
}

/*void expr2whilet::convert_overflow(const exprt &src, unsigned &precedence)
{
  precedence = 16;

  std::string dest = "overflow(\"";
  dest += src.id().c_str() + 9;
  dest += "\"";

  forall_operands(it, src)
  {
    unsigned p;
    std::string arg_str = convert(*it, p);

    dest += ", ";
    // TODO: [add] brackets, if necessary, depending on p
    dest += arg_str;
  }

  dest += ")";

  return dest;
}*/

std::string expr2whilet::indent_str(unsigned indent)
{
  std::string dest;
  for(unsigned j = 0; j < indent; j++)
    dest += ' ';
  return dest;
}

/*void expr2whilet::convert_code_asm(const codet &, unsigned indent)
{
  std::string dest = indent_str(indent);
  dest += "asm();\n";
  return dest;
}*/

void expr2whilet::convert_code_while(const codet &src, unsigned indent)
{
  if(src.operands().size() != 2)
  {
    log_error("incorrect number of operands for while loop");
  }

  rapid_file << indent_str(indent);
  rapid_file << "while(";

  convert(src.op0());

  rapid_file << "){\n";

  convert_code(to_code(src.op1()), indent + 2);
  
  rapid_file << indent_str(indent);
  rapid_file << "}\n";
}

void expr2whilet::convert_code_dowhile(const codet &src, unsigned indent)
{
  if(src.operands().size() != 2)
    throw norep_error("incorrect number of operands for do while loop");

  // In which cases can it be nil???
  if(!src.op1().is_nil()){
    // bring thw loop body outside loop
    convert_code(to_code(src.op1()), indent);
  }

  rapid_file << indent_str(indent);
  rapid_file << "while(";

  convert(src.op0());

  rapid_file << "){\n";

  if(!src.op1().is_nil()){
    // bring thw loop body outside loop
    convert_code(to_code(src.op1()), indent + 2);
  } else {
    rapid_file << indent_str(indent + 2);
    rapid_file << "skip;\n";
  }

  rapid_file << indent_str(indent);
  rapid_file << "}\n";
}

void expr2whilet::convert_code_ifthenelse(const codet &src, unsigned indent)
{
  if(src.operands().size() != 3 && src.operands().size() != 2)
    throw norep_error("incorrect number of operands for if else statement");


  rapid_file << indent_str(indent);
  rapid_file << "if(";

  convert(src.op0());

  rapid_file << "){\n";

  if(src.op1().is_nil())
  {
    rapid_file << indent_str(indent + 2);
    rapid_file << "skip;\n";
  } else {
    convert_code(to_code(src.op1()), indent + 2);
  }

  rapid_file << indent_str(indent);
  rapid_file << "} else {\n";

  if(src.operands().size() == 3 && !src.operands().back().is_nil())
  {
    convert_code(to_code(src.operands().back()), indent + 2);
  } else {
    rapid_file << indent_str(indent + 2);
    rapid_file << "skip;\n";    
  }

  rapid_file << indent_str(indent);  
  rapid_file << "}\n";
}


void expr2whilet::convert_code_switch(const codet &src, unsigned indent)
{
  /*if(src.operands().size() < 1)
  {
    unsigned precedence;
    return convert_norep(src, precedence);
  }

  std::string dest = indent_str(indent);
  dest += "switch(";
  dest += convert(src.op0());
  dest += ")\n";

  dest += indent_str(indent);
  dest += "{\n";

  for(unsigned i = 1; i < src.operands().size(); i++)
  {
    const exprt &op = src.operands()[i];

    if(op.statement() != "block")
    {
      unsigned precedence;
      dest += convert_norep(op, precedence);
    }
    else
    {
      forall_operands(it, op)
        dest += convert_code(to_code(*it), indent + 2);
    }
  }

  dest += "\n";
  dest += indent_str(indent);
  dest += '}';*/

  //return dest;
}

void expr2whilet::convert_code_decl_block(const codet &src, unsigned indent)
{
  forall_operands(it, src)
  {
    convert_code(to_code(*it), indent);
  }
}

void expr2whilet::convert_code_dead(const codet &src, unsigned indent)
{
  // initializer to go away
  if(src.operands().size() != 1)
  {
    unsigned precedence;
 //   return convert_norep(src, precedence);
  }

  //return indent_str(indent) + "dead " + convert(src.op0()) + ";";
}

void expr2whilet::convert_code_decl(const codet &src, unsigned indent)
{

  if(src.operands().size() != 1 && src.operands().size() != 2)
  {
    log_error("could not convert decl to While as incorrect number of operands");
  }

  rapid_file << indent_str(indent);

  // TODO const qualifier?
  convert(src.op0().type());

  rapid_file << " ";

  convert(src.op0());

  if(src.operands().size() == 2){

    rapid_file << " = ";

    convert(src.op1());

  }

  rapid_file << ";\n";
  //return dest;
}

void expr2whilet::convert_code_for(const codet &src, unsigned indent)
{
  if(src.operands().size() != 4)
    throw norep_error("incorrect number of operands for for loop");

  if(!src.operands()[0].is_nil()){
    convert_code(to_code(src.operands()[0]), indent);
  }

  rapid_file << indent_str(indent);
  rapid_file << "while(";

  if(!src.operands()[1].is_nil())
  {
    convert(src.operands()[1]);
  } else {
    // wierdly Rapid does not suppurt true and false as Boolean statements
    // currently
    rapid_file << "2 == 2";
  }

  rapid_file << "){\n";

  if(src.op3().is_nil()){
    rapid_file << indent_str(indent + 2);
    rapid_file << "skip;\n";
  }
  else
  {
    convert_code(to_code(src.op3()), indent + 2);
  }

  if(!src.op2().is_nil())
  {
    convert_code(to_code(src.operands()[2]), indent + 2);
  }

  rapid_file << indent_str(indent);
  rapid_file << "}\n";
}

void expr2whilet::convert_code_block(const codet &src, unsigned indent)
{

  forall_operands(it, src)
  {
    if(it->statement() == "block")
      // TODO While has no support for code blocks
      convert_code_block(to_code(*it), indent + 2);
    else
      convert_code(to_code(*it), indent);
  }
}

void expr2whilet::convert_code_expression(const codet &src, unsigned indent)
{

  if(src.operands().size() != 1)
  {
    log_error("Expression has unsupported number of operands");
  }
  
  auto op0 = src.op0();

  rapid_file << indent_str(indent);

  if(op0.id() == "sideeffect")
  {
    const irep_idt &statement = op0.statement();

    unsigned precedence;
    if(statement == "postincrement")
      convert_unary_post(op0, "++", precedence = 16);
    else if(statement == "postdecrement")
      convert_unary_post(op0, "--", precedence = 16);
    else if(statement == "assign+")
      convert_binary(op0, "+=", precedence = 2, true);
    else if(statement == "assign-")
      convert_binary(op0, "-=", precedence = 2, true);
    else if(statement == "assign*")
      convert_binary(op0, "*=", precedence = 2, true);
    else if(statement == "assign_mod")
      convert_binary(op0, "%=", precedence = 2, true);
    else if(statement == "function_call")
      convert_function_call(op0, precedence);
    else if(statement == "malloc")
      convert_malloc(op0, precedence = 15);
    else
      convert(op0);
    //  return convert_norep(src, precedence);
  }

  rapid_file << ";\n";

  //return dest;
}

void expr2whilet::convert_main(const codet &src)
{
  rapid_file << "func main() {\n";

  // start with indent of 2
  convert_code(src, 2);

  rapid_file << "}";
}


void expr2whilet::convert_code(const codet &src, unsigned indent)
{
  const irep_idt &statement = src.statement();

  //printf(statement.c_str());
  //printf("\n");
 
  if(statement == "expression")
    convert_code_expression(src, indent);

  if(statement == "block")
    convert_code_block(src, indent);

  if(statement == "switch")
    convert_code_switch(src, indent);

  if(statement == "for")
    convert_code_for(src, indent);

  if(statement == "while")
    convert_code_while(src, indent);

//  if(statement == "asm")
//    convert_code_asm(src, indent);

  if(statement == "skip")
    indent_str(indent) + "skip;\n";

  if(statement == "dowhile")
    convert_code_dowhile(src, indent);

  if(statement == "ifthenelse")
    convert_code_ifthenelse(src, indent);

  if(statement == "return")
    throw unsupported_error("Rapid does not yet support return statements");

  if(statement == "goto")
    throw unsupported_error("Rapid does not yet support goto statements");

  if(statement == "gcc_goto")
    throw unsupported_error("Rapid does not yet support goto statements");

  if(statement == "printf")
  { // do nothing. assue that printf is sideeffect free, so we just ignore. presto!
  }

  if(statement == "assume")
    convert_code_assume(src, indent);

  if(statement == "assert")
    convert_code_assert(src, indent);

  if(statement == "break")
    throw unsupported_error("Rapid does not yet support break statements");

  if(statement == "continue")
    throw unsupported_error("Rapid does not yet support continue statements");

  if(statement == "decl")
    convert_code_decl(src, indent);

  if(statement == "decl-block")
    convert_code_decl_block(src, indent);

  if(statement == "dead")
    convert_code_dead(src, indent);

  if(statement == "assign")
    convert_code_assign(src, indent);

  if(statement == "init")
    convert_code_init(src, indent);

  if(statement == "lock")
    convert_code_lock(src, indent);

  if(statement == "unlock")
    convert_code_unlock(src, indent);

  if(statement == "function_call")
    throw unsupported_error("Rapid does not yet support function calls. Try running with option --full-inlining");

  if(statement == "label")
    convert_code_label(to_code_label(src), indent);

  if(statement == "switch_case")
    convert_code_switch_case(to_code_switch_case(src), indent);

  if(statement == "free")
    convert_code_free(src, indent);

  unsigned precedence;
  //return convert_norep(src, precedence);
}

void expr2whilet::convert_code_assign(const codet &src, unsigned indent)
{
  // Union remangle: If the right hand side is a constant array, containing
  // byte extract expressions, then it's almost 100% certain to be a flattened
  // union literal. Precise identification isn't feasible right now, sadly.
  // In that case, replace with a special intrinsic indicating to the user that
  // the original code is now meaningless.
  
  /*unsigned int precedent = 15;
  std::string tmp = convert(src.op0(), precedent);
  tmp += "=";
  tmp += convert(src.op1(), precedent);

  std::string dest = indent_str(indent) + tmp + ";";*/

  //return dest;
}

void expr2whilet::convert_code_free(const codet &src, unsigned indent)
{
  if(src.operands().size() != 1)
  {
    unsigned precedence;
  //  return convert_norep(src, precedence);
  }

  //return indent_str(indent) + "FREE(" + convert(src.op0()) + ");";
}

void expr2whilet::convert_code_init(const codet &src, unsigned indent)
{
  //std::string tmp = convert_binary(src, "=", 2, true);

  //return indent_str(indent) + "INIT " + tmp + ";";
}

void expr2whilet::convert_code_lock(const codet &src, unsigned indent)
{
  if(src.operands().size() != 1)
  {
    unsigned precedence;
  //  return convert_norep(src, precedence);
  }

  //return indent_str(indent) + "LOCK(" + convert(src.op0()) + ");";
}

void expr2whilet::convert_code_unlock(const codet &src, unsigned indent)
{
  if(src.operands().size() != 1)
  {
    unsigned precedence;
  //  return convert_norep(src, precedence);
  }

  //return indent_str(indent) + "UNLOCK(" + convert(src.op0()) + ");";
}


void expr2whilet::convert_code_assert(const codet &src, unsigned indent)
{
  if(src.operands().size() != 1)
  {
    unsigned precedence;
  //  return convert_norep(src, precedence);
  }

  //return indent_str(indent) + "assert(" + convert(src.op0()) + ");";
}

void expr2whilet::convert_code_assume(const codet &src, unsigned indent)
{
  if(src.operands().size() != 1)
  {
    unsigned precedence;
  //  return convert_norep(src, precedence);
  }

  //return indent_str(indent) + "assume(" + convert(src.op0()) + ");";
}

void expr2whilet::convert_code_label(const code_labelt &src, unsigned indent)
{
 /* std::string labels_string;

  irep_idt label = src.get_label();

  labels_string += "\n";
  labels_string += indent_str(indent);
  labels_string += name2string(label);
  labels_string += ":\n";

  std::string tmp = convert_code(src.code(), indent + 2);*/

  //return labels_string + tmp;
}

void
expr2whilet::convert_code_switch_case(const code_switch_caset &src, unsigned indent)
{
  /*std::string labels_string;

  if(src.is_default())
  {
    labels_string += "\n";
    labels_string += indent_str(indent);
    labels_string += "default:\n";
  }
  else
  {
    labels_string += "\n";
    labels_string += indent_str(indent);
    labels_string += "case ";
    labels_string += convert(src.case_op());
    labels_string += ":\n";
  }

  unsigned next_indent = indent;
  if(
    src.code().get_statement() != "block" &&
    src.code().get_statement() != "switch_case")
    next_indent += 2;
  std::string tmp = convert_code(src.code(), next_indent);*/

  //return labels_string + tmp;
}

/*void expr2whilet::convert_code(const codet &src)
{
  return convert_code(src, 0);
}



/* Checks whether the expression `e` is one performing pointer-arithmetic, that
 * is, addition/subtraction of an integer-typed expression to/from a
 * pointer-typed expression.
 *
 * If so, `true` is returned and `ptr` holds the address of the inner-most
 * pointer-typed expression while `idx` gets assigned a (newly constructed, in
 * case of multiple levels of pointer-typed expressions) expression that
 * corresponds to the index into `*ptr`.
 *
 * Note, just a pointer-typed symbol (or constant) is not recognized as pointer-
 * arithmetic.
 */
static bool is_pointer_arithmetic(const exprt &e, const exprt *&ptr, exprt &idx)
{
  if(e.type().id() != "pointer")
    return false;

  ptr = &e;

  /* a pointer-typed arithmetic (+ or -) expression cannot be unary in the C
   * language */
  assert(!(e.id() == "unary+" || (e.id() == "+" && e.operands().size() == 1)));
  assert(!(e.id() == "unary-" || (e.id() == "-" && e.operands().size() == 1)));

  if(e.id() == "+" || e.id() == "-")
  {
    assert(e.operands().size() == 2);
    const exprt *p = nullptr, *i = nullptr;
    auto categorize = [&p, &i](const exprt &e) {
      const irep_idt &tid = e.type().id();
      if(tid == "pointer")
        p = &e;
      else if(tid == "signedbv" || tid == "unsignedbv")
        i = &e;
    };
    categorize(e.op0());
    categorize(e.op1());
    if(p && i)
    {
      if(e.id() == "-")
        assert(i == &e.op1());
      exprt j;
      if(is_pointer_arithmetic(*p, p, j))
      {
        auto is_unsigned = [](const exprt &e) {
          return e.type().id() == "unsignedbv";
        };
        const char *type =
          is_unsigned(j) || is_unsigned(*i) ? "unsignedbv" : "signedbv";
        idx = exprt(e.id(), typet(type));
        idx.copy_to_operands(j, *i);
      }
      else if(e.id() == "-")
      {
        idx = exprt("unary-", i->type());
        idx.copy_to_operands(*i);
      }
      else
        idx = *i;
      ptr = p;
      return true;
    }
  }

  return false;
}

void expr2whilet::convert(const exprt &src, unsigned &precedence)
{
  precedence = 16;

  if(src.id() == "+")
    convert_binary(src, "+", precedence = 12, false);

  if(src.id() == "-")
  {
    convert_binary(src, "-", precedence = 12, true);
  }

  else if(src.id() == "unary-")
  {
    convert_unary(src, "-", precedence = 15);
  }

  else if(src.id() == "unary+")
  {
    convert_unary(src, "+", precedence = 15);
  }

  /*else if(src.id() == "invalid-pointer")
  {
    return convert_function(src, "INVALID-POINTER", precedence = 15);
  }

  else if(src.id() == "invalid-object")
  {
    return "invalid-object";
  }*/

  else if(src.id() == "NULL-object")
  {
    //return "0";
  }

 /* else if(src.id() == "infinity")
  {
    return convert_function(src, "INFINITY", precedence = 15);
  }*/

  else if(src.id() == "builtin-function")
  {
    //return src.identifier().as_string();
  }

  else if(src.id() == "pointer_object")
  {
    //return convert_function(src, "POINTER_OBJECT", precedence = 15);
  }

  else if(src.id() == "object_value")
  {
    //return convert_function(src, "OBJECT_VALUE", precedence = 15);
  }

  /*else if(src.id() == "pointer_object_has_type")
  {
    return convert_pointer_object_has_type(src, precedence = 15);
  }

  else if(src.id() == "array_of")
  {
    return convert_array_of(src, precedence = 15);
  }*/

  else if(src.id() == "pointer_offset")
  {
    //return convert_function(src, "POINTER_OFFSET", precedence = 15);
  }

  else if(src.id() == "pointer_base")
  {
    //return convert_function(src, "POINTER_BASE", precedence = 15);
  }

  else if(src.id() == "pointer_cons")
  {
    //return convert_function(src, "POINTER_CONS", precedence = 15);
  }

  else if(src.id() == "same-object")
  {
    //return convert_function(src, "SAME-OBJECT", precedence = 15);
  }

  else if(src.id() == "valid_object")
  {
   // return convert_function(src, "VALID_OBJECT", precedence = 15);
  }

  else if(src.id() == "deallocated_object" || src.id() == "memory-leak")
  {
    //return convert_function(src, "DEALLOCATED_OBJECT", precedence = 15);
  }

  else if(src.id() == "dynamic_object")
  {
    //return convert_function(src, "DYNAMIC_OBJECT", precedence = 15);
  }

  else if(src.id() == "is_dynamic_object")
  {
    //return convert_function(src, "IS_DYNAMIC_OBJECT", precedence = 15);
  }

  else if(src.id() == "dynamic_size")
  {
    //return convert_function(src, "DYNAMIC_SIZE", precedence = 15);
  }

  else if(src.id() == "dynamic_type")
  {
    //return convert_function(src, "DYNAMIC_TYPE", precedence = 15);
  }

  else if(src.id() == "pointer_offset")
  {
    //return convert_function(src, "POINTER_OFFSET", precedence = 15);
  }

  else if(src.id() == "isnan")
  {
    //return convert_function(src, "isnan", precedence = 15);
  }

  else if(src.id() == "isfinite")
  {
    //return convert_function(src, "isfinite", precedence = 15);
  }

  else if(src.id() == "isinf")
  {
    //return convert_function(src, "isinf", precedence = 15);
  }

  else if(src.id() == "isnormal")
  {
    //return convert_function(src, "isnormal", precedence = 15);
  }

  else if(src.id() == "signbit")
  {
    //return convert_function(src, "signbit", precedence = 15);
  }

  else if(src.id() == "nearbyint")
  {
    //return convert_function(src, "nearbyint", precedence = 15);
  }

  else if(src.id() == "popcount")
  {
    //return convert_function(src, "popcount", precedence = 15);
  }

  else if(src.id() == "bswap")
  {
    //return convert_function(src, "bswap", precedence = 15);
  }

  else if(src.id() == "builtin_va_arg")
  {
    //return convert_function(src, "builtin_va_arg", precedence = 15);
  }

  else if(src.is_address_of())
  {
    if(src.op0().id() == "label")
      throw unsupported_error("Rapid does not support expression " + expr2c(src.op0(), ns));

    return convert_unary(src, "&", precedence = 15);
  }

  else if(src.id() == "dereference")
  {
    //if(src.operands().size() != 1)
    //  return convert_norep(src, precedence);

    /* Special case for `*(p+i)` and `*(p-i)`: these constructs could either
     * already be in the source or it was created artificially by the frontend
     * for `p[i]`, see clang_c_adjust::adjust_index() and also Github issue
     * #725. As those expressions are semantically indistinguishable and also
     * supported by verifiers, choose to print the succint form `p[i]` in all
     * cases. */
    exprt idx;
    const exprt *ptr;
    if(is_pointer_arithmetic(src.op0(), ptr, idx))
    {
      // TODO potentially treat this as an array ?
      throw unsupported_error("Rapid cannot handle pointer arithmetic");
    }

    convert_unary(src, "*", precedence = 15);
  }

  else if(src.id() == "index")
    convert_index(src, precedence = 16);

  else if(src.id() == "member")
    convert_member(src, precedence = 16);

  /*else if(src.id() == "array-member-value")
    return convert_array_member_value(src, precedence = 16);

  else if(src.id() == "struct-member-value")
    return convert_struct_member_value(src, precedence = 16);*/

  else if(src.id() == "sideeffect")
  {
    const irep_idt &statement = src.statement();

    if(statement == "preincrement"){
      throw unsupported_error("Rapid does not yet support the use of pre- postfix operators as expressions " + expr2c(src, ns));
    }
    if(statement == "predecrement"){
      throw unsupported_error("Rapid does not yet support the use of pre- postfix operators as expressions " + expr2c(src, ns));
    }
    else if(statement == "postincrement"){
      throw unsupported_error("Rapid does not yet support the use of pre- postfix operators as expressions " + expr2c(src, ns));
    }
    else if(statement == "postdecrement"){
      throw unsupported_error("Rapid does not yet support the use of pre- postfix operators as expressions " + expr2c(src, ns));
    }
    else if(statement == "assign+"){
      throw unsupported_error("Rapid does not yet support the use of assignment as an expression " + expr2c(src, ns));
    }
    else if(statement == "assign-"){
      throw unsupported_error("Rapid does not yet support the use of assignment as an expression " + expr2c(src, ns));
    }
    else if(statement == "assign*"){
      throw unsupported_error("Rapid does not yet support the use of assignment as an expression " + expr2c(src, ns));
    }
    else if(statement == "assign_div"){
      throw unsupported_error("Rapid does not yet support the use of assignment as an expression " + expr2c(src, ns));
    }
    else if(statement == "assign_mod"){
      throw unsupported_error("Rapid does not yet support the use of assignment as an expression " + expr2c(src, ns));
    }
    else if(statement == "assign"){
      convert_binary(src, "=", precedence = 2, true);
    }
    else if(statement == "function_call"){
      convert_function_call(src, precedence);
    }
    else if(statement == "malloc")
      convert_malloc(src, precedence = 15);
    else if(statement == "printf")
      convert_function(src, "PRINTF", precedence = 15);
    else if(statement == "nondet")
      convert_nondet(src, precedence = 15);
  /*  else if(statement == "statement_expression")
      return convert_statement_expression(src, precedence = 15);*/
    else if(statement == "va_arg")
      convert_function(src, "va_arg", precedence = 15);
    else
      printf("garbage");
    //  return convert_norep(src, precedence);
  }

  else if(src.id() == "not")
    convert_unary(src, "!", precedence = 15);

  else if(src.id() == "*")
    convert_binary(src, src.id_string(), precedence = 13, false);

  else if(src.id() == "/")
    convert_binary(src, src.id_string(), precedence = 13, true);

  else if(src.id() == "mod")
    return convert_binary(src, "%", precedence = 13, true);

  /*else if(src.id() == "shl")
    return convert_binary(src, "<<", precedence = 11, true);

  else if(src.id() == "ashr" || src.id() == "lshr")
    return convert_binary(src, ">>", precedence = 11, true);*/

  else if(
    src.id() == "<" || src.id() == ">" || src.id() == "<=" || src.id() == ">=")
    convert_binary(src, src.id_string(), precedence = 10, true);

  else if(src.id() == "notequal")
    convert_binary(src, "!=", precedence = 9, true);

  else if(src.id() == "=")
    convert_binary(src, "==", precedence = 9, true);

  /*else if(src.id() == "ieee_add")
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
    return convert_function(src, "abs", precedence = 15);*/

  else if(src.is_and())
    convert_binary(src, "&&", precedence = 5, false);

  else if(src.id() == "or")
    convert_binary(src, "||", precedence = 4, false);

  else if(src.id() == "=>")
    convert_binary(src, "=>", precedence = 3, true);

  else if(src.id() == "if")
    convert_trinary(src, "?", ":", precedence = 3);

 /* 
  else if(src.id() == "with")
    return convert_with(src, precedence = 2);*/

  else if(src.id() == "symbol")
    return convert_symbol(src, precedence);

 /* else if(src.id() == "nondet_symbol")
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
    return convert_object_descriptor(src, precedence);*/

  else if(src.is_code())
    convert_code(to_code(src), 0);

  else if(src.id() == "constant")
    convert_constant(src, precedence);

  else if(src.id() == "string-constant")
    convert_constant(src, precedence);

  else if(src.id() == "struct")
    convert_struct(src, precedence);

  //else if(src.id() == "union")
  //  return convert_union(src, precedence);

  //else if(src.is_array())
  //  convert_array(src, precedence);

  //else if(src.id() == "array-list")
  //  return convert_array_list(src, precedence);

  else if(src.id() == "typecast")
    return convert_typecast(src, precedence);

  //else if(src.id() == "bitcast")
  //  return convert_bitcast(src, precedence);

 // else if(src.id() == "implicit_address_of")
 //   return convert_implicit_address_of(src, precedence);

  //else if(src.id() == "implicit_dereference")
  //  return convert_function(src, "IMPLICIT_DEREFERENCE", precedence = 15);

  else if(src.id() == "comma")
    convert_binary(src, ", ", precedence = 1, false);

  //else if(src.id() == "cond")
  //  return convert_cond(src, precedence);

  //else if(std::string(src.id_string(), 0, 9) == "overflow-")
  //  return convert_overflow(src, precedence);

  //else if(src.id() == "unknown")
  //  return "*";

  //else if(src.id() == "invalid")
  //  return "#";

  /*else if(src.id() == "extractbit")
    return convert_extractbit(src, precedence);

  else if(src.id() == "sizeof")
    return convert_sizeof(src, precedence);*/

  //else if(src.id() == "concat")
  //  return convert_function(src, "CONCAT", precedence = 15);

  //else if(src.id() == "extract")
  //  return convert_extract(src);

  // no C language expression for internal representation
  //return convert_norep(src, precedence);
}

void expr2whilet::convert(const exprt &src)
{
  unsigned precedence;
  convert(src, precedence);
}

/*std::string expr2c(const exprt &expr, const namespacet &ns, bool fullname)
{
  std::string code;
  expr2ct expr2c(ns, fullname);
  expr2c.get_shorthands(expr);
  return expr2c.convert(expr);
}

std::string type2c(const typet &type, const namespacet &ns, bool fullname)
{
  expr2ct expr2c(ns, fullname);
  return expr2c.convert(type);
}*/
