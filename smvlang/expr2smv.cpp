/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <string.h>

#include <lispexpr.h>
#include <lispirep.h>
#include <bitvector.h>
#include <i2string.h>

#include "expr2smv.h"

class expr2smvt
{
public:
  bool convert_nondet_choice(const exprt &src, std::string &dest,
                             unsigned precedence);

  bool convert_binary(const exprt &src, std::string &dest, const std::string &symbol,
                      unsigned precedence);

  bool convert_unary(const exprt &src, std::string &dest, const std::string &symbol,
                     unsigned precedence);

  bool convert_index(const exprt &src, std::string &dest, unsigned precedence);

  bool convert(const exprt &src, std::string &dest, unsigned &precedence);

  bool convert(const exprt &src, std::string &dest);

  bool convert_symbol(const exprt &src, std::string &dest, unsigned &precedence);

  bool convert_next_symbol(const exprt &src, std::string &dest, unsigned &precedence);

  bool convert_constant(const exprt &src, std::string &dest, unsigned &precedence);

  bool convert_cond(const exprt &src, std::string &dest);

  bool convert_norep(const exprt &src, std::string &dest, unsigned &precedence);

  bool convert(const typet &src, std::string &dest);
};

/* SMV Operator Precedences:

 1 %left  COMMA
 2 %right IMPLIES
 3 %left  IFF
 4 %left  OR XOR XNOR
 5 %left  AND
 6 %left  NOT
 7 %left  EX AX EF AF EG AG EE AA SINCE UNTIL TRIGGERED RELEASES EBF EBG ABF ABG BUNTIL MMIN MMAX
 8 %left  OP_NEXT OP_GLOBAL OP_FUTURE
 9 %left  OP_PREC OP_NOTPRECNOT OP_HISTORICAL OP_ONCE
10 %left  APATH EPATH
11 %left  EQUAL NOTEQUAL LT GT LE GE
12 %left  UNION
13 %left  SETIN
14 %left  MOD
15 %left  PLUS MINUS
16 %left  TIMES DIVIDE
17 %left  UMINUS
18 %left  NEXT SMALLINIT
19 %left  DOT
20        [ ]
21 = max

*/

#define SMV_MAX_PRECEDENCE 21

/*******************************************************************\

Function: expr2smvt::convert_nondet_choice

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2smvt::convert_nondet_choice(
  const exprt &src,
  std::string &dest,
  unsigned precedence)
{
  dest="{ ";

  bool first=true;

  forall_operands(it, src)
  {
    if(first)
      first=false;
    else
      dest+=", ";

    std::string tmp;
    if(convert(*it, tmp)) return true;
    dest+=tmp;
  }

  dest+=" }";
  return false;
}

/*******************************************************************\

Function: expr2smvt::convert_cond

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2smvt::convert_cond(
  const exprt &src,
  std::string &dest)
{
  dest="case ";

  bool condition=true;

  forall_operands(it, src)
  {
    std::string tmp;
    if(convert(*it, tmp)) return true;
    dest+=tmp;

    if(condition)
      dest+=": ";
    else
      dest+="; ";
    
    condition=!condition;
  }

  dest+="esac";
  return false;
}

/*******************************************************************\

Function: expr2smvt::convert_binary

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2smvt::convert_binary(
  const exprt &src,
  std::string &dest,
  const std::string &symbol,
  unsigned precedence)
{
  if(src.operands().size()<2)
    return convert_norep(src, dest, precedence);

  bool first=true;

  forall_operands(it, src)
  {
    if(first)
      first=false;
    else
    {
      dest+=' ';
      dest+=symbol;
      dest+=' ';
    }

    std::string op;
    unsigned p;

    if(convert(*it, op, p)) return true;

    if(precedence>p) dest+='(';
    dest+=op;
    if(precedence>p) dest+=')';
  }

  return false;
}

/*******************************************************************\

Function: expr2smvt::convert_unary

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2smvt::convert_unary(
  const exprt &src, std::string &dest,
  const std::string &symbol,
  unsigned precedence)
{
  if(src.operands().size()!=1)
    return convert_norep(src, dest, precedence);
    
  std::string op;
  unsigned p;

  if(convert(src.op0(), op, p)) return true;

  dest+=symbol;
  if(precedence>p) dest+='(';
  dest+=op;
  if(precedence>p) dest+=')';

  return false;
}

/*******************************************************************\

Function: expr2smvt::convert_index

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2smvt::convert_index(
  const exprt &src,
  std::string &dest,
  unsigned precedence)
{
  if(src.operands().size()!=2)
    return convert_norep(src, dest, precedence);

  std::string op;
  unsigned p;

  if(convert(src.op0(), op, p)) return true;

  if(precedence>p) dest+='(';
  dest+=op;
  if(precedence>p) dest+=')';

  if(convert(src.op1(), op, p)) return true;

  dest+='[';
  dest+=op;
  dest+=']';

  return false;
}

/*******************************************************************\

Function: expr2smvt::convert_norep

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2smvt::convert_norep(
  const exprt &src,
  std::string &dest,
  unsigned &precedence)
{
  precedence=SMV_MAX_PRECEDENCE;
  dest=src.to_string();
  return false;
}

/*******************************************************************\

Function: expr2smvt::convert_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2smvt::convert_symbol(
  const exprt &src,
  std::string &dest,
  unsigned &precedence)
{
  precedence=SMV_MAX_PRECEDENCE;
  dest=src.get_string("identifier");
 
  if(strncmp(dest.c_str(), "smv::", 5)==0)
    dest.erase(0, 5);

  return false;
}

/*******************************************************************\

Function: expr2smvt::convert_next_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2smvt::convert_next_symbol(
  const exprt &src,
  std::string &dest,
  unsigned &precedence)
{
  std::string tmp;
  convert_symbol(src, tmp, precedence);

  dest="next("+tmp+")";

  return false;
}

/*******************************************************************\

Function: expr2smvt::convert_constant

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2smvt::convert_constant(
  const exprt &src,
  std::string &dest,
  unsigned &precedence)
{
  precedence=SMV_MAX_PRECEDENCE;

  const typet &type=src.type();
  const std::string &value=src.get_string("value");

  if(type.id()=="bool")
  {
    if(src.is_true())
      dest+="1";
    else
      dest+="0";
  }
  else if(type.id()=="integer" ||
          type.id()=="natural" ||
          type.id()=="range")
    dest=value;
  else
    return convert_norep(src, dest, precedence);

  return false;
}

/*******************************************************************\

Function: expr2smvt::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2smvt::convert(
  const exprt &src,
  std::string &dest,
  unsigned &precedence)
{
  precedence=SMV_MAX_PRECEDENCE;

  if(src.id()=="+")
    return convert_binary(src, dest, "+", precedence=15);

  else if(src.id()=="-")
  {
    if(src.operands().size()<2)
      return convert_norep(src, dest, precedence);
    else     
      return convert_binary(src, dest, "-", precedence=15);
  }

  else if(src.id()=="unary-")
  {
    if(src.operands().size()!=1)
      return convert_norep(src, dest, precedence);
    else     
      return convert_unary(src, dest, "-", precedence=17);
  }

  else if(src.id()=="index")
    return convert_index(src, dest, precedence=20);

  else if(src.id()=="*" || src.id()=="/")
    return convert_binary(src, dest, src.id_string(), precedence=16);

  else if(src.id()=="<" || src.id()==">" ||
          src.id()=="<=" || src.id()==">=")
    return convert_binary(src, dest, src.id_string(), precedence=11);

  else if(src.id()=="=")
    return convert_binary(src, dest, "=", precedence=11);

  else if(src.id()=="not")
    return convert_unary(src, dest, "!", precedence=6);

  else if(src.id()=="and")
    return convert_binary(src, dest, "&", precedence=5);

  else if(src.id()=="or")
    return convert_binary(src, dest, "|", precedence=4);

  else if(src.id()=="=>")
    return convert_binary(src, dest, "->", precedence=2);

  else if(src.id()=="<=>")
    return convert_binary(src, dest, "<->", precedence=3);

  else if(src.id()=="AG" || src.id()=="EG" ||
          src.id()=="AX" || src.id()=="EX")
    return convert_unary(src, dest, src.id_string()+" ", precedence=7);

  else if(src.id()=="symbol")
    return convert_symbol(src, dest, precedence);

  else if(src.id()=="next_symbol")
    return convert_next_symbol(src, dest, precedence);

  else if(src.id()=="constant")
    return convert_constant(src, dest, precedence);

  else if(src.id()=="smv_nondet_choice")
    return convert_nondet_choice(src, dest, precedence);

  else if(src.id()=="cond")
    return convert_cond(src, dest);

  else // no SMV language expression for internal representation 
    return convert_norep(src, dest, precedence);

  return false;
}

/*******************************************************************\

Function: expr2smvt::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2smvt::convert(const exprt &src, std::string &dest)
{
  unsigned precedence;
  return convert(src, dest, precedence);
}

/*******************************************************************\

Function: expr2smv

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2smv(const exprt &expr, std::string &code)
{
  expr2smvt expr2smv;
  return expr2smv.convert(expr, code);
}

/*******************************************************************\

Function: type2smv

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool type2smv(const typet &type, std::string &code)
{
  if(type.id()=="bool")
    code="boolean";
  else if(type.id()=="array")
  {
    std::string tmp;
    if(type2smv(type.subtype(), tmp)) return true;
    code="array ";
    code+="..";
    code+=" of ";
    code+=tmp;
  }
  else if(type.id()=="enum")
  {
    const irept &elements=type.find("elements");
    code="{ ";
    bool first=true;
    forall_irep(it, elements.get_sub())
    {
      if(first) first=false; else code+=", ";
      code+=it->id_string();
    }
    code+=" }";
  }
  else if(type.id()=="range")
  {
    code=type.get_string("from")+".."+type.get_string("to");
  }
  else if(type.id()=="submodule")
  {
    code=type.get_string("identifier");
    const exprt &e=(exprt &)type;
    if(e.has_operands())
    {
      code+='(';
      bool first=true;
      forall_operands(it, e)
      {
        if(first) first=false; else code+=", ";
        std::string tmp;
        expr2smv(*it, tmp);
        code+=tmp;
      }
      code+=')';
    }
  }
  else
    return true;

  return false;
}
