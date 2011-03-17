/*******************************************************************\

Module: Conversion of Expressions

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <iostream>
#include <string>

#include <arith_tools.h>
#include <config.h>
#include <rename.h>
#include <bitvector.h>
#include <c_misc.h>
#include <prefix.h>
#include <cprover_prefix.h>
#include <i2string.h>

#include "convert-c.h"
#include "c_typecast.h"
#include "c_sizeof.h"
#include "c_types.h"
#include "convert_float_literal.h"

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const CharConstant &expression,
  exprt &dest)
{
  if(expression.wide)
  {
    err_location(expression);
    err << "error: wide char not supported: " << expression;
    throw 0;
  }

  typet type(config.ansi_c.char_is_unsigned?"unsignedbv":"signedbv");
  type.set("width", config.ansi_c.char_width);
  dest=from_integer(expression.ch, type);
  
  std::string cformat;
  cformat="'";

  #ifdef HAS_original_rep
  if(expression.original_rep=="")
    MetaChar(cformat, expression.ch, false);
  else
    cformat+=expression.original_rep;
  #else
  MetaChar(cformat, expression.ch, false);
  #endif

  cformat+="'";  
  dest.set("#cformat", cformat);
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const IntConstant &expression,
  exprt &dest)
{
  dest.type()=typet("signedbv");

  dest.type().set("width",
    expression.L?config.ansi_c.long_int_width:config.ansi_c.int_width);

  mp_integer int_value((signed long long int)expression.lng);
  from_integer(int_value, dest);

  #ifdef HAS_original_rep
  if(expression.original_rep!="")
    dest.set("#cformat", expression.original_rep);
  #endif
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const UIntConstant &expression,
  exprt &dest)
{
  dest.type()=typet("unsignedbv");

  dest.type().set("width",
    expression.L?config.ansi_c.long_int_width:config.ansi_c.int_width);

  mp_integer int_value((unsigned long long int)expression.ulng);
  from_integer(int_value, dest);

  #ifdef HAS_original_rep
  if(expression.original_rep!="")
    dest.set("#cformat", expression.original_rep);
  #endif
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const FloatConstant &expression,
  exprt &dest)
{
  convert_float_literal(expression.value, dest);
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const StringConstant &expression,
  exprt &dest)
{
  if(expression.wide)
  {
    err_location(expression);
    err << "error: wide string not supported: " << expression;
    throw 0;
  }

  exprt size=from_integer(expression.buff.size()+1, int_type());

  dest.id("string-constant");
  dest.type()=typet("array");
  dest.type().subtype()=char_type();
  dest.type().set("size", size);

  dest.set("value", expression.buff);
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const ArrayConstant &expression,
  exprt &dest)
{
  dest.id("constant");
  dest.type()=typet("incomplete_array");

  dest.reserve_operands(expression.items.size());

  for(ExprVector::const_iterator it=expression.items.begin();
      it!=expression.items.end(); it++)
  {
    exprt item;
    convert(**it, item);
    dest.move_to_operands(item);
  }
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const EnumConstant &expression,
  exprt &dest)
{
  // never used?
  err_location(expression);
  err << "error: enum constant not yet supported: " << expression;
  throw 0;
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const Constant &expression,
  exprt &dest)
{
  switch(expression.ctype)
  {
   case CT_Char:   convert((CharConstant &)expression,   dest); break;
   case CT_Int:    convert((IntConstant &)expression,    dest); break;
   case CT_UInt:   convert((UIntConstant &)expression,   dest); break;
   case CT_Float:  convert((FloatConstant &)expression,  dest); break;
   case CT_String: convert((StringConstant &)expression, dest); break;
   case CT_Array:  convert((ArrayConstant &)expression,  dest); break;
   case CT_Enum:   convert((EnumConstant &)expression,   dest); break;

   default:
    err_location(expression);
    err << "error: unknown constant: " << expression;
    throw 0;
  }
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const SymEntry *entry,
  const Location &location,
  irept &dest)
{
  // lookup in symbol map

  std::map<const SymEntry *, std::string>::const_iterator it=
    symbol_map.find(entry);

  if(it==symbol_map.end())
  {
    err_location(location);
    err << "error: symbol \"" << entry->name
        << "\" not found";
    throw 0;
  }

  dest=exprt("symbol");
  dest.set("identifier", it->second);
  set_location(dest, location);
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const symbolt &symbol,
  const locationt &location,
  exprt &dest)
{
  if(symbol.is_macro)
    dest=symbol.value;
  else if(has_prefix(symbol.name, CPROVER_PREFIX "constant_infinity"))
  {
    dest.id("infinity");
    dest.type()=symbol.type;
    dest.location()=location;
  }
  else
  {
    dest.id("symbol");
    dest.type()=symbol.type;
    dest.set("identifier", symbol.name);
    dest.location()=location;

    if(symbol.lvalue) dest.set("#lvalue", true);
    if(symbol.is_static) dest.set("#static", true);
    if(symbol.is_extern) dest.set("#extern", true);
    if(symbol.is_volatile) dest.set("#volatile", true);

    if(dest.type().id()=="code") // function designator
    { // special case: this is sugar for &f
      exprt tmp("address_of");
      tmp.set("#implicit", true);
      tmp.type().id("pointer");
      tmp.type().subtype()=dest.type();
      tmp.move_to_operands(dest);
      dest.swap(tmp);
    }
  }
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_enum(
  const EnumDef &def,
  const std::string &name,
  mp_integer &offset)
{
  int i;

  // search for element

  for(i=0; i<def.nElements; i++)
    if(name==def.names[i]->name)
      break;

  // found?

  if(i==def.nElements)
  {
    err << "error: enum constant not found: `" << name << "'";
    throw 0;
  }

  // found!

  offset=0;

  #if 0
  while(true)
  {
    if(def.values[i]!=NULL)
    {
      exprt tmp;
      convert(*def.values[i], tmp);

      make_constant_index(tmp);

      mp_integer int_value;

      if(to_integer(tmp, int_value))
      {
        err_location(*def.values[i]);
        err << "failed to convert enum value: "
            << to_string(tmp);
        throw 0;
      }

      offset+=int_value;
      break;
    }

    if(i==0) break;
    offset=offset+1;
    i--;
  }
  #endif
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const Variable &expression,
  exprt &dest)
{
  if(expression.name==NULL)
  {
    err_location(expression);
    err << "error: variable without name: " << expression;
    throw 0;
  }

  if(expression.name->entry==NULL)
  {
    // we don't have it
    err_location(expression);
    err << "variable `" << expression.name->name << "' not declared";
    throw 0;
  }

  if(expression.name->entry->IsEnumConst())
  {
    // really an integer constant, not a variable
    mp_integer offset;
    const EnumDef &def=*expression.name->entry->u2EnumDef;

    convert_enum(def, expression.name->entry->name, offset);
    dest=from_integer(offset, enum_type());
    dest.set("#cformat", expression.name->entry->name);
    return;
  }

  if(!expression.name->entry->IsVarDecl() &&
     !expression.name->entry->IsFctDecl() &&
     !expression.name->entry->IsParamDecl())
  {
    err_location(expression);
    err << "error: not a variable and not a function: " << expression;
    throw 0;
  }

  convert(expression.name->entry, expression.location, dest);
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const FunctionCall &expression,
  exprt &dest)
{
  dest.id("sideeffect");
  dest.set("statement", "functioncall");
 
  exprt new_expr;
  convert(*expression.function, new_expr);

  dest.reserve_operands(expression.args.size()+1);
  
  dest.move_to_operands(new_expr);

  for(unsigned i=0; i<expression.args.size(); i++)
  {
    exprt op;
    convert(*expression.args[i], op);
    dest.move_to_operands(op);
  }
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const UnaryExpr &expression,
  exprt &dest)
{
  dest.operands().push_back((exprt &)nil_rep);
  exprt &operand=dest.operands().back();
  convert(*expression.operand(), operand);

  //const typet &type0=dest.op0().type();

  switch(expression.op())
  {
  case UO_Plus:    // +
  case UO_Minus:   // -
    dest.id((expression.op()==UO_Plus)?"+":"unary-");
    return;

  case UO_BitNot:  // ~
    dest.id("bitnot");
    return;

  case UO_Not:     // !
    dest.id("not");
    return;

  case UO_PreInc:  dest.id("sideeffect"); dest.set("statement", "preincrement");  return; // ++x
  case UO_PreDec:  dest.id("sideeffect"); dest.set("statement", "predecrement");  return; // --x
  case UO_PostInc: dest.id("sideeffect"); dest.set("statement", "postincrement"); return; // x++
  case UO_PostDec: dest.id("sideeffect"); dest.set("statement", "postdecrement"); return; // x--

  case UO_AddrOf:  // &
    dest.id("address_of");
    return;

  case UO_Deref:   // *
    dest.id("dereference");
    return;

  default:
    break;
  }

  err_location(expression);
  err << "error: failed to convert operator `";
  PrintUnaryOp(err, expression.op());
  err << "'";
  throw 0;
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const AssignExpr &expression,
  exprt &dest)
{
  std::string op;

  switch(expression.op())
  {
   case AO_Equal:     op="assign";        break; // =
   case AO_PlusEql:   op="assign+";       break; // +=
   case AO_MinusEql:  op="assign-";       break; // -=
   case AO_MultEql:   op="assign*";       break; // *=
   case AO_DivEql:    op="assign_div";    break; // /=
   case AO_ModEql:    op="assign_mod";    break; // %=
   case AO_ShlEql:    op="assign_shl";    break; // <<=
   case AO_ShrEql:    op="assign_shr";    break; // >>= 
   case AO_BitAndEql: op="assign_bitand"; break; // &=
   case AO_BitXorEql: op="assign_bitxor"; break; // ^=
   case AO_BitOrEql:  op="assign_bitor";  break; // |=

   default:
    err_location(expression);
    err << "error: unknown assign expression: " << expression;
    throw 0;
  }

  dest.id("sideeffect");
  dest.set("statement", op);
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const BinaryExpr &expression,
  exprt &dest)
{
  if(expression.op()==BO_Member ||
     expression.op()==BO_PtrMember)
  {
    dest.operands().resize(1);

    convert(*expression.leftExpr(), dest.op0());

    if(expression.rightExpr()->etype==ET_Variable)
    {
      const Variable &op1=(const Variable &)*expression.rightExpr();

      if(op1.name==NULL)
      {
        err_location(expression);
        throw "expected flield name, but got NULL name";
      }
      
      dest.set("component_name", op1.name->name);
    }
    else
    {
      err_location(expression);
      throw "expected flield name";
    }
  }
  else
  {
    dest.operands().resize(2);

    convert(*expression.leftExpr(), dest.op0());
    convert(*expression.rightExpr(), dest.op1());
  }

  std::string op;

  switch(expression.op())
  {
  case BO_Plus:   op="+";      break; // +
  case BO_Minus:  op="-";      break; // -
  case BO_Mult:   op="*";      break; // *
  case BO_Div:    op="/";      break; // /
  case BO_Mod:    op="mod";    break; // %

  case BO_Shl:    op="shl";    break; // <<
  case BO_Shr:    op="shr";    break; // >>
  case BO_BitAnd: op="bitand"; break; // &
  case BO_BitXor: op="bitxor"; break; // ^
  case BO_BitOr:  op="bitor";  break; // |

  case BO_And:    op="and";    break; // &&
  case BO_Or:     op="or";     break; // ||
  #ifdef HAS_BO_Implies
  case BO_Implies:op="=>";     break; // =>
  #endif

  case BO_Comma:  op="comma";  break; // x,y
  case BO_Member: op="member"; break; // x.y
  case BO_PtrMember: op="ptrmember"; break; // x->y
  case BO_Assign:    // An AssignExpr
    return convert((const AssignExpr &)expression, dest);

  case BO_Rel:       // A RelExpr
    return convert((const RelExpr &)expression, dest);

  default:
    err_location(expression);
    err << "error: unknown binary expression: " << expression;
    throw 0;
  }
  
  dest.id(op);
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const RelExpr &expression,
  exprt &dest)
{
  std::string op;

  switch(expression.op())
  {
   case RO_Equal:    op="=";        break; // ==
   case RO_NotEqual: op="notequal"; break; // !=
   case RO_Less:     op="<";        break; // <
   case RO_LessEql:  op="<=";       break; // <=
   case RO_Grtr:     op=">";        break; // >
   case RO_GrtrEql:  op=">=";       break; // >=

   default:
    err_location(expression);
    err << "error: unknown binary expression: " << expression;
    throw 0;
  }

  dest.id(op);
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const TrinaryExpr &expression,
  exprt &dest)
{
  exprt::operandst &operands=dest.operands();

  operands.resize(3);
  
  exprt &op0=operands.front();
  exprt &op1=*(++operands.begin());
  exprt &op2=operands.back();

  convert(*expression.condExpr(),  op0);
  convert(*expression.trueExpr(),  op1);
  convert(*expression.falseExpr(), op2);

  dest.id("if");
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const IndexExpr &expression,
  exprt &dest)
{
  exprt index_expr, array_expr;

  dest.id("index");
  
  convert(*expression._subscript, index_expr);
  convert(*expression.array, array_expr);
  
  dest.move_to_operands(array_expr, index_expr);
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const SizeofExpr &expression,
  exprt &dest)
{
  dest.id("sizeof");

  if(expression.sizeofType!=NULL)
  {
    typet &sizeof_type=(typet &)dest.add("sizeof-type");
    convert(*expression.sizeofType, expression.location, sizeof_type);
  }
  else if(expression.expr!=NULL)
  {
    exprt expr;
    convert(*expression.expr, expr);
    dest.move_to_operands(expr);
  }
  else
  {
    err << "sizeof without parameter: " << expression;
    throw 0;
  }
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

#ifdef HAS_QuantExpr
void convert_ct::convert(
  const QuantExpr &expression,
  exprt &dest)
{
  exprt sub, ident;

  convert(*expression.sub, sub);
  convert(expression.ident->entry, expression.location, ident);

  // renaming

  const symbolt *symbol;

  if(lookup(ident.get("identifier"), symbol)) 
    return true;
  
  symbolt new_symbol=get_new_name(*symbol, context);

  add_symbol(new_symbol);

  rename(sub, symbol->name, new_symbol.name);
  ident.set("identifier", new_symbol.name);

  // build quantifier expression

  if(expression.qOp==Q_Forall)
    dest.id("forall");
  else
    dest.id("exists");

  implicit_typecast_bool(sub);

  dest.type()=typet("bool");

  dest.move_to_operands(ident);
  dest.move_to_operands(sub);
}
#endif

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const CastExpr &expression,
  exprt &dest)
{
  exprt expr;

  convert(*expression.expr, expr);
  convert(*expression.castTo, expression.location, dest.type());

  dest.id("typecast");
  dest.move_to_operands(expr);
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const Expression &expression,
  exprt &dest)
{
  dest.make_nil();

  dest.add("type");

  switch(expression.etype)
  {
   case ET_VoidExpr:
    err_location(expression);
    err << "error: void expression: " << expression;
    throw 0;

   case ET_Constant:     convert((Constant &)expression, dest); break;
   case ET_Variable:     convert((Variable &)expression, dest); break;
   case ET_FunctionCall: convert((FunctionCall &) expression, dest); break;
   case ET_UnaryExpr:    convert((UnaryExpr &)expression, dest); break;
   case ET_BinaryExpr:   convert((BinaryExpr &)expression, dest); break;
   case ET_TrinaryExpr:  convert((TrinaryExpr &)expression, dest); break;
   case ET_IndexExpr:    convert((IndexExpr &)expression, dest); break;
   case ET_SizeofExpr:   convert((SizeofExpr &)expression, dest); break;
   #ifdef HAS_QuantExpr
   case ET_QuantExpr:    convert((QuantExpr &)expression, dest); break;
   #endif
   case ET_CastExpr:     convert((CastExpr &)expression, dest); break;
   
   default:
    err_location(expression);
    err << "error: unknown expression: " << expression;
    throw 0;
  }

  set_location(dest, expression.location);
}

/*******************************************************************\

Function: convert_ct::err_location

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::err_location(const Location &location)
{
  if(location.line==0 || location.column==0) return;

  locationt irep_location;
  convert_location(irep_location, location);
  
  error_handlert::err_location(irep_location);
}
