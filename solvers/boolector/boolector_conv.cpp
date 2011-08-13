/*******************************************************************
 Module:

 Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

 \*******************************************************************/

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include<sstream>
#include <arith_tools.h>
#include <std_types.h>
#include <config.h>
#include <i2string.h>
#include <expr_util.h>
#include <prefix.h>
#include <string2array.h>
#include <pointer_offset_size.h>
#include <find_symbols.h>
#include <fixedbv.h>
#include <solvers/flattening/boolbv_width.h>

#include "boolector_conv.h"
#include "../ansi-c/c_types.h"

extern "C" {
#include <boolector.h>
}

//#define DEBUG

boolector_convt::boolector_convt(std::ostream &_out) :
        boolector_prop_wrappert(_out),
        prop_convt(boolector_prop)
{

  number_variables_boolector=0;
  set_to_counter=0;
  boolector_prop.boolector_ctx = boolector_new();
  boolector_ctx = boolector_prop.boolector_ctx;
  boolector_enable_model_gen(boolector_ctx);
  //boolector_enable_inc_usage(boolector_ctx);
  //btorFile = fopen ( "btor.txt" , "wb" );
  //smtFile = fopen ( "smt.txt" , "wb" );
}

boolector_convt::~boolector_convt()
{

  //fclose(btorFile);
  //fclose(smtFile);
  if (boolector_prop.btor && boolector_prop.assumpt.size()>0)
  {
    btorFile = fopen ( filename.c_str() , "wb" );

    for(unsigned i=0; i<boolector_prop.assumpt.size(); i++)
      boolector_dump_smt(boolector_ctx, btorFile, boolector_prop.assumpt.at(i));

    fclose (btorFile);
  }
}

/*******************************************************************
 Function: boolector_convt::print_data_types

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void boolector_convt::print_data_types(BtorExp* operand0, BtorExp* operand1)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}

/*******************************************************************
 Function: boolector_convt::check_all_types

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::check_all_types(const typet &type)
{
  if (type.is_bool() || type.is_signedbv() || type.is_unsignedbv() ||
	  type.is_symbol() || type.is_empty() || type.is_fixedbv() ||
	  type.is_array() || type.is_struct() || type.is_pointer() ||
	  type.is_union())
  {
    return true;
  }

  return false;
}

/*******************************************************************
 Function: boolector_convt::is_signed

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::is_signed(const typet &type)
{
  if (type.is_signedbv() || type.is_fixedbv())
    return true;

  return false;
}

/*******************************************************************
 Function: boolector_convt::check_boolector_properties

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

int boolector_convt::check_boolector_properties(void)
{
  return boolector_sat(boolector_ctx);
}

/*******************************************************************
 Function: boolector_convt::is_ptr

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::is_ptr(const typet &type)
{
  return type.is_pointer() || type.id()=="reference";
}

/*******************************************************************
 Function: boolector_convt::convert_pointer_offset

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_pointer_offset(unsigned bits, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}
/*******************************************************************
 Function: boolector_convt::select_pointer_value

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::select_pointer_value(BtorExp* object, BtorExp* offset, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}

/*******************************************************************
 Function: boolector_convt::set_filename

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void boolector_convt::set_filename(std::string file)
{
  filename = file;
}

/*******************************************************************
 Function: boolector_convt::create_boolector_array

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::create_boolector_array(const typet &type, std::string identifier, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  BtorExp *array;
  unsigned int width = 0;

  if (type.subtype().is_bool())
  {
    array = boolector_array(boolector_ctx, 1, config.ansi_c.int_width, identifier.c_str());
  }
  else if (type.subtype().is_fixedbv())
  {
	width = atoi(type.subtype().width().c_str());
	array = boolector_array(boolector_ctx, width, config.ansi_c.int_width, identifier.c_str());
  }
  else if (type.subtype().is_signedbv() || type.subtype().is_unsignedbv())
  {
	width = atoi(type.subtype().width().c_str());
	array = boolector_array(boolector_ctx, width, config.ansi_c.int_width, identifier.c_str());
  }
  else if (type.subtype().is_pointer())
  {
	create_boolector_array(type.subtype(), identifier, array);
  }

  bv = array;

  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_identifier

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_identifier(const std::string &identifier, const typet &type, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "identifier: " << identifier << std::endl;
  std::cout << "type.pretty(): " << type.pretty() << std::endl;
#endif

  unsigned int width = 0;

  width = atoi(type.width().c_str());

  if (type.is_bool())
  {
	bv = boolector_var(boolector_ctx, 1, identifier.c_str());
  }
  else if (type.is_signedbv() || type.is_unsignedbv() || type.id()=="c_enum")
  {
	bv = boolector_var(boolector_ctx, width, identifier.c_str());
  }
  else if (type.is_fixedbv())
  {
	bv = boolector_var(boolector_ctx, width, identifier.c_str());
  }
  else if (type.is_array())
  {
	create_boolector_array(type, identifier, bv);
  }
  else if (type.is_pointer())
  {
	if (convert_identifier(identifier, type.subtype(), bv))
	  return true;
  }

#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif
  return false;
}

/*******************************************************************\

Function: boolector_convt::convert_bv
  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool boolector_convt::convert_bv(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif
#if 1
  bv_cachet::const_iterator cache_result=bv_cache.find(expr);
  if(cache_result!=bv_cache.end())
  {
#ifdef DEBUG
    std::cout << "Cache hit on " << expr.pretty() << "\n";
#endif
	bv = cache_result->second;
    return false;
  }
#endif

#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  if (convert_boolector_expr(expr, bv))
    return true;

#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  // insert into cache
  bv_cache.insert(std::pair<const exprt, BtorExp*>(expr, bv));

  return false;
}

/*******************************************************************
 Function: boolector_convt::read_cache

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::read_cache(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  std::string symbol;
  unsigned int size = pointer_cache.size();

  symbol = expr.identifier().as_string();

  for(pointer_cachet::const_iterator it = pointer_cache.begin();
  it != pointer_cache.end(); it++)
  {
	if (symbol.compare((*it).second.c_str())==0)
	{
	  //std::cout << "Cache hit on: " << (*it).first.pretty() << "\n";
	  if (convert_bv((*it).first, bv))
	    return true;
	  else
	    return false;
	}
  }

  return convert_bv(expr, bv);
}

/*******************************************************************
 Function: boolector_convt::write_cache

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void boolector_convt::write_cache(const exprt &expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  std::string symbol, identifier;

  identifier = expr.identifier().as_string();

  for (std::string::const_iterator it = identifier.begin(); it
		!= identifier.end(); it++)
  {
	char ch = *it;

	if (isalnum(ch) || ch == '$' || ch == '?')
	{
	  symbol += ch;
	}
	else if (ch == '#')
	{
      pointer_cache.insert(std::pair<const exprt, std::string>(expr, symbol));
      return;
	}
  }
}

/*******************************************************************
 Function: boolector_convt::convert_lt

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

BtorExp* boolector_convt::convert_lt(const exprt &expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()==2);
  BtorExp *constraint, *operand0, *operand1;

  if (expr.op0().type().is_array())
    write_cache(expr.op0());

  if (convert_bv(expr.op0(), operand0))
    return boolector_false(boolector_ctx);
  if (convert_bv(expr.op1(), operand1))
	return boolector_false(boolector_ctx);

  if (expr.op1().type().is_signedbv() || expr.op1().type().is_fixedbv())
	constraint = boolector_slt(boolector_ctx, operand0, operand1);
  else if (expr.op1().type().is_unsignedbv())
	constraint = boolector_ult(boolector_ctx, operand0, operand1);

  return constraint;
}

/*******************************************************************
 Function: boolector_convt::convert_gt

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

BtorExp* boolector_convt::convert_gt(const exprt &expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()==2);
  BtorExp *constraint, *operand0, *operand1;

  if (expr.op0().type().is_array())
    write_cache(expr.op0());

  if (convert_bv(expr.op0(), operand0))
    return boolector_false(boolector_ctx);
  if (convert_bv(expr.op1(), operand1))
	return boolector_false(boolector_ctx);

  if (expr.op1().type().is_signedbv() || expr.op1().type().is_fixedbv())
	constraint = boolector_sgt(boolector_ctx, operand0, operand1);
  else if (expr.op1().type().is_unsignedbv())
	constraint = boolector_ugt(boolector_ctx, operand0, operand1);

  return constraint;
}


/*******************************************************************
 Function: boolector_convt::convert_le

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

BtorExp* boolector_convt::convert_le(const exprt &expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()==2);
  BtorExp *constraint, *operand0, *operand1;

  if (expr.op0().type().is_array())
    write_cache(expr.op0());

  if (convert_bv(expr.op0(), operand0))
	  return boolector_false(boolector_ctx);
  if (convert_bv(expr.op1(), operand1))
	  return boolector_false(boolector_ctx);

  if (expr.op1().type().is_signedbv() || expr.op1().type().is_fixedbv())
	constraint = boolector_slte(boolector_ctx, operand0, operand1);
  else if (expr.op1().type().is_unsignedbv())
	constraint = boolector_ulte(boolector_ctx, operand0, operand1);

  return constraint;
}

/*******************************************************************
 Function: boolector_convt::convert_ge

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

BtorExp* boolector_convt::convert_ge(const exprt &expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()==2);
  BtorExp *constraint, *operand0, *operand1;

  if (expr.op0().type().is_array())
    write_cache(expr.op0());

  if (convert_bv(expr.op0(), operand0))
	  return boolector_false(boolector_ctx);
  if (convert_bv(expr.op1(), operand1))
	  return boolector_false(boolector_ctx);

  //std::cout << expr.pretty() << std::endl;

  if (expr.op1().type().is_signedbv() || expr.op1().type().is_fixedbv())
	constraint = boolector_sgte(boolector_ctx, operand0, operand1);
  else if (expr.op1().type().is_unsignedbv())
	constraint = boolector_ugte(boolector_ctx, operand0, operand1);

  //std::cout << expr.pretty() << std::endl;
  //if (expr.op0().is_symbol() && expr.op1().is_constant())
    //boolector_assert(boolector_ctx, constraint);

  return constraint;
}


/*******************************************************************
 Function: boolector_convt::convert_eq

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

BtorExp* boolector_convt::convert_eq(const exprt &expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "expr.pretty(): " << expr.pretty() << std::endl;
#endif

  assert(expr.operands().size()==2);
  static BtorExp *constraint, *operand0, *operand1;

  if (expr.op0().type().is_array())
    write_cache(expr.op0());

  if (convert_bv(expr.op0(), operand0))
	  return boolector_false(boolector_ctx);
  if (convert_bv(expr.op1(),operand1))
	  return boolector_false(boolector_ctx);

  if (expr.id() == "=")
	constraint = boolector_eq(boolector_ctx, operand0, operand1);
  else
  {
	constraint = boolector_ne(boolector_ctx, operand0, operand1);
	//std::cout << "expr.op1().is_constant(): " << expr.op1().is_constant() << std::endl;
	//std::cout << "expr.pretty(): " << expr.pretty() << std::endl;
	//if (expr.op0().is_symbol() && expr.op1().is_constant())
	  //boolector_assert(boolector_ctx, constraint);
  }


  return constraint;
}

/*******************************************************************
 Function: boolector_convt::convert_invalid_pointer

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

BtorExp* boolector_convt::convert_invalid(const exprt &expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}

/*******************************************************************
 Function: boolector_convt::convert_same_object

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

BtorExp* boolector_convt::convert_same_object(const exprt &expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}

/*******************************************************************
 Function: boolector_convt::convert_dynamic_object

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

BtorExp* boolector_convt::convert_dynamic_object(const exprt &expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}

/*******************************************************************
 Function: boolector_convt::convert_overflow_sum

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

BtorExp* boolector_convt::convert_overflow_sum(const exprt &expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()==2);
  static BtorExp *bv, *operand[2];

  if (convert_bv(expr.op0(), operand[0]))
	  return boolector_false(boolector_ctx);
  if (convert_bv(expr.op1(), operand[1]))
	  return boolector_false(boolector_ctx);

  if (expr.op0().type().is_signedbv() && expr.op1().type().id()=="signedbv")
    bv = boolector_saddo(boolector_ctx, operand[0], operand[1]);
  else if (expr.op0().type().is_unsignedbv() && expr.op1().type().is_unsignedbv())
	bv = boolector_uaddo(boolector_ctx, operand[0], operand[1]);

  return bv; //boolector_not(boolector_ctx, bv);
}

/*******************************************************************
 Function: boolector_convt::convert_overflow_sub

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

BtorExp* boolector_convt::convert_overflow_sub(const exprt &expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()==2);
  static BtorExp *bv, *operand[2];

  if (convert_bv(expr.op0(), operand[0]))
	  return boolector_false(boolector_ctx);
  if (convert_bv(expr.op1(), operand[1]))
	  return boolector_false(boolector_ctx);

  if (expr.op0().type().is_signedbv() && expr.op1().type().id()=="signedbv")
    bv = boolector_ssubo(boolector_ctx, operand[0], operand[1]);
  else if (expr.op0().type().is_unsignedbv() && expr.op1().type().is_unsignedbv())
	bv = boolector_usubo(boolector_ctx, operand[0], operand[1]);

  return bv; //boolector_not(boolector_ctx, bv);
}

/*******************************************************************
 Function: boolector_convt::convert_overflow_mul

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

BtorExp* boolector_convt::convert_overflow_mul(const exprt &expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()==2);
  static BtorExp *bv, *operand[2];

  if (convert_bv(expr.op0(), operand[0]))
	  return boolector_false(boolector_ctx);
  if (convert_bv(expr.op1(), operand[1]))
	  return boolector_false(boolector_ctx);

  if (expr.op0().type().is_signedbv() && expr.op1().type().id()=="signedbv")
    bv = boolector_smulo(boolector_ctx, operand[0], operand[1]);
  else if (expr.op0().type().is_unsignedbv() && expr.op1().type().is_unsignedbv())
	bv = boolector_umulo(boolector_ctx, operand[0], operand[1]);

  return bv; //boolector_not(boolector_ctx, bv);
}

/*******************************************************************
 Function: boolector_convt::convert_overflow_unary

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

BtorExp* boolector_convt::convert_overflow_unary(const exprt &expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "expr.pretty(): " << expr.pretty() << std::endl;
  std::cout << "width: " << width << std::endl;
#endif

  assert(expr.operands().size()==1);
  static BtorExp *bv, *operand;
  u_int i, width;



  if (convert_bv(expr.op0(), operand))
	  return boolector_false(boolector_ctx);

  if (boolbv_get_width(expr.op0().type(), width))
	  return boolector_false(boolector_ctx);

  bv = boolector_not(boolector_ctx, boolector_ne(boolector_ctx, operand, boolector_ones(boolector_ctx,width)));

#if 0
  if (expr.op0().type().is_signedbv())
    bv = boolector_slt(boolector_ctx, boolector_neg(boolector_ctx, operand), boolector_ones(boolector_ctx,width));
  else if (expr.op0().type().is_unsignedbv())
	bv = boolector_ult(boolector_ctx, boolector_neg(boolector_ctx, operand), boolector_ones(boolector_ctx,width));
#endif

  return bv;
}

/*******************************************************************
 Function: boolector_convt::convert_overflow_typecast

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

BtorExp* boolector_convt::convert_overflow_typecast(const exprt &expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  unsigned bits=atoi(expr.id().c_str()+18);

  const exprt::operandst &operands=expr.operands();

  if(operands.size()!=1)
    throw "operand "+expr.id_string()+" takes one operand";

  static BtorExp *bv, *operand[3], *mid, *overflow[2], *tmp, *minus_one, *two;
  u_int i, result=1, width;
  std::string value;

  boolbv_get_width(expr.op0().type(), width);

  if(bits>=width || bits==0)
    throw "overflow-typecast got wrong number of bits";

  assert(bits <= 32);

  for(i=0; i<bits; i++)
  {
	if (i==31)
	  result=(result-1)*2+1;
	else if (i<31)
      result*=2;
  }

  if (is_signed(expr.op0().type()))
    value = integer2string(binary2integer(expr.op0().value().as_string(), true),10);
  else
	value = integer2string(binary2integer(expr.op0().value().as_string(), false),10);

  if (convert_bv(expr.op0(), operand[0]))
	  return boolector_false(boolector_ctx);

  if (expr.op0().type().is_signedbv() || expr.op0().type().is_fixedbv())
  {
	tmp = boolector_int(boolector_ctx, result, width);
	two = boolector_int(boolector_ctx, 2, width);
	minus_one = boolector_int(boolector_ctx, -1, width);
	mid = boolector_sdiv(boolector_ctx, tmp, two);
	operand[1] = boolector_sub(boolector_ctx, mid, minus_one);
	operand[2] = boolector_mul(boolector_ctx, operand[1], minus_one);

	overflow[0] = boolector_slt(boolector_ctx, operand[0], operand[1]);
	overflow[1] = boolector_sgt(boolector_ctx, operand[0], operand[2]);
	bv = boolector_not(boolector_ctx, boolector_and(boolector_ctx, overflow[0], overflow[1]));
  }
  else if (expr.op0().type().is_unsignedbv())
  {
	operand[2] = boolector_unsigned_int(boolector_ctx, 0, width);
	operand[1] = boolector_unsigned_int(boolector_ctx, result, width);
	overflow[0] = boolector_ult(boolector_ctx, operand[0], operand[1]);
	overflow[1] = boolector_ugt(boolector_ctx, operand[0], operand[2]);
	bv = boolector_not(boolector_ctx, boolector_and(boolector_ctx, overflow[0], overflow[1]));
  }

  return bv;
}

/*******************************************************************
 Function: boolector_convt::convert_rest

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

literalt boolector_convt::convert_rest(const exprt &expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "expr.pretty(): " << expr.pretty() << "\n";
#endif

  literalt l = boolector_prop.new_variable();
  static BtorExp *constraint, *formula;

  if (!assign_boolector_expr(expr))
	return l;

  if (expr.id() == "=" || expr.is_notequal())
	constraint = convert_eq(expr);
  else if (expr.id() == "<")
	constraint = convert_lt(expr);
  else if (expr.id() == ">")
	constraint = convert_gt(expr);
  else if (expr.id() == "<=")
	constraint = convert_le(expr);
  else if (expr.id() == ">=")
	constraint = convert_ge(expr);
  else if (expr.id() == "overflow-+")
	constraint = convert_overflow_sum(expr);
  else if (expr.id() == "overflow--")
	constraint = convert_overflow_sub(expr);
  else if (expr.id() == "overflow-*")
	constraint = convert_overflow_mul(expr);
  else if (expr.id() == "overflow-unary-")
	constraint = convert_overflow_unary(expr);
  else if(has_prefix(expr.id_string(), "overflow-typecast-"))
	constraint = convert_overflow_typecast(expr);
  else
	throw "convert_boolector_expr: " + expr.id_string() + " is not supported yet";

#ifdef DEBUG
  std::cout << "convert_rest l" << l.var_no() << std::endl;
#endif

  formula = boolector_iff(boolector_ctx, boolector_prop.boolector_literal(l), constraint);
  boolector_assert(boolector_ctx, formula);

  if (boolector_prop.btor)
    boolector_prop.assumpt.push_back(formula);

  //boolector_dump_btor(boolector_ctx, btorFile, formula);
  //boolector_dump_smt(boolector_ctx, smtFile, formula);

  return l;
}

/*******************************************************************
 Function: boolector_convt::convert_rel

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_rel(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "expr.pretty(): " << expr.pretty() << std::endl;
#endif

  assert(expr.operands().size()==2);

  BtorExp *result, *operand0, *operand1;

  if (convert_bv(expr.op0(), operand0)) return true;
  if (convert_bv(expr.op1(), operand1)) return true;

  const typet &op_type=expr.op0().type();

  if (op_type.is_unsignedbv() || op_type.subtype().is_unsignedbv())
  {
    if(expr.id()=="<=")
   	  result = boolector_ulte(boolector_ctx,operand0,operand1);
    else if(expr.id()=="<")
      result = boolector_ult(boolector_ctx,operand0,operand1);
    else if(expr.id()==">=")
      result = boolector_ugt(boolector_ctx,operand0,operand1);
    else if(expr.id()==">")
      result = boolector_ugte(boolector_ctx,operand0,operand1);
  }
  else if (op_type.is_signedbv() || op_type.is_fixedbv() ||
			 op_type.subtype().is_signedbv() || op_type.subtype().is_fixedbv() )
  {
    if(expr.id()=="<=")
      result = boolector_slte(boolector_ctx,operand0,operand1);
    else if(expr.id()=="<")
      result = boolector_ult(boolector_ctx,operand0,operand1);
    else if(expr.id()==">=")
      result = boolector_sgt(boolector_ctx,operand0,operand1);
    else if(expr.id()==">")
      result = boolector_sgte(boolector_ctx,operand0,operand1);
  }

  bv = result;

  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_typecast

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_typecast(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "expr.pretty(): " << expr.pretty() << std::endl;
#endif

  assert(expr.operands().size()==1);

  BtorExp *result, *operand;
  BtorExp *args[2];

  const exprt &op=expr.op0();

  if(expr.type().is_signedbv() || expr.type().is_unsignedbv())
  {
    unsigned to_width=atoi(expr.type().width().c_str());

    if(op.type().is_signedbv())
    {
      unsigned from_width=atoi(op.type().width().c_str());
      //std::cout << "from_width: " << from_width << "\n";
      //std::cout << "to_width: " << to_width << "\n";

      if(from_width==to_width)
      {
    	if (convert_bv(op, result))
    	  return true;
      }
      else if(from_width<to_width)
      {
    	if (convert_bv(op, operand))
    	  return true;
    	result = boolector_sext(boolector_ctx, operand, (to_width-from_width));
      }
      else if (from_width>to_width)
      {
    	if (convert_bv(op, operand))
    	  return true;
    	result = boolector_slice(boolector_ctx, operand, (to_width-1), 0);
      }
    }
    else if(op.type().is_unsignedbv())
    {
      unsigned from_width=atoi(op.type().width().c_str());

      if(from_width==to_width)
      {
    	if (convert_bv(op, result))
    	  return true;
      }
      else if(from_width<to_width)
      {
    	if (convert_bv(op, operand))
    	  return true;
    	result = boolector_uext(boolector_ctx, operand, (to_width-from_width));
      }
      else if (from_width>to_width)
      {
      	if (convert_bv(op, operand))
      	  return true;
      	result = boolector_slice(boolector_ctx, operand, (to_width-1), 0);
      }
    }
    else if (op.type().is_bool())
    {
  	  BtorExp *zero, *one;
      unsigned width;

      boolbv_get_width(expr.type(), width);
  	  if (expr.type().is_signedbv() || expr.type().is_fixedbv())
  	  {
  	    zero = boolector_int(boolector_ctx, 0, width);
  	    one =  boolector_int(boolector_ctx, 1, width);
  	  }
  	  else if (expr.type().is_unsignedbv())
  	  {
  	    zero = boolector_unsigned_int(boolector_ctx, 0, width);
  	    one =  boolector_unsigned_int(boolector_ctx, 1, width);
  	  }

  	  if (convert_bv(op,operand)) return true;
  	  result = boolector_cond(boolector_ctx, operand, one, zero);
    }
    if (op.type().is_pointer())
    {
      unsigned width;
      unsigned object=pointer_logic.add_object(expr);
      boolbv_get_width(expr.type(), width);
      result = boolector_int(boolector_ctx, object, width);
    }
  }
  else if(expr.type().is_fixedbv())
  {
    const fixedbv_typet &fixedbv_type=to_fixedbv_type(expr.type());
    unsigned to_fraction_bits=fixedbv_type.get_fraction_bits();
    unsigned to_integer_bits=fixedbv_type.get_integer_bits();

    if(op.type().is_unsignedbv() ||
       op.type().is_signedbv() ||
       op.type().id()=="enum")
    {
      unsigned from_width;

   	  boolbv_get_width(op.type(), from_width);

      if(from_width==to_integer_bits)
      {
    	if (convert_bv(op, result)) return true;
      }
      else if(from_width>to_integer_bits)
      {
    	if (convert_bv(op, args[0])) return true;
      	result = boolector_slice(boolector_ctx, args[0], (from_width-1), to_integer_bits);
      }
      else
      {
        assert(from_width<to_integer_bits);

        if(expr.type().is_unsignedbv())
        {
       	  if (convert_bv(op, args[0])) return true;
          result = boolector_uext(boolector_ctx, args[0], (to_integer_bits-from_width));
        }
        else
        {
          if (convert_bv(op, args[0])) return true;
      	  result = boolector_sext(boolector_ctx, args[0], (to_integer_bits-from_width));
        }
      }

      result = boolector_concat(boolector_ctx, result, boolector_int(boolector_ctx, 0, to_fraction_bits));
    }
    else if(op.type().is_bool())
    {
      BtorExp *zero, *one;
      unsigned width;

      boolbv_get_width(expr.type(), width);

  	  zero = boolector_int(boolector_ctx, 0, to_integer_bits);
  	  one = boolector_int(boolector_ctx, 1, to_integer_bits);
  	  result = boolector_cond(boolector_ctx, result, one, zero);
  	  result = boolector_concat(boolector_ctx, result, boolector_int(boolector_ctx, 0, to_fraction_bits));
    }
    else if(op.type().is_fixedbv())
    {
      BtorExp *magnitude, *fraction;
      const fixedbv_typet &from_fixedbv_type=to_fixedbv_type(op.type());
      unsigned from_fraction_bits=from_fixedbv_type.get_fraction_bits();
      unsigned from_integer_bits=from_fixedbv_type.get_integer_bits();
      unsigned from_width=from_fixedbv_type.get_width();

      if(to_integer_bits<=from_integer_bits)
      {
    	if (convert_bv(op, args[0])) return true;
        magnitude = boolector_slice(boolector_ctx, args[0], (from_fraction_bits+to_integer_bits-1), from_fraction_bits);
      }
      else
      {
        assert(to_integer_bits>from_integer_bits);
        if (convert_bv(op, args[0])) return true;
        magnitude = boolector_sext(boolector_ctx, boolector_slice(boolector_ctx, args[0], from_width-1, from_fraction_bits), (to_integer_bits-from_integer_bits));
      }

      if(to_fraction_bits<=from_fraction_bits)
      {
        if (convert_bv(op, args[0])) return true;
        fraction = boolector_slice(boolector_ctx, args[0], (from_fraction_bits-1), from_fraction_bits-to_fraction_bits);
      }
      else
      {
        assert(to_fraction_bits>from_fraction_bits);

        if (convert_bv(op, args[0])) return true;
        fraction = boolector_concat(boolector_ctx, boolector_slice(boolector_ctx, args[0], (from_fraction_bits-1), 0), boolector_int(boolector_ctx, 0, to_fraction_bits-from_fraction_bits));
      }

      result = boolector_concat(boolector_ctx, magnitude, fraction);
    }
    else
      throw "unexpected typecast to fixedbv";
  }

  if(expr.type().id()=="c_enum")
  {
	BtorExp *zero, *one;
	unsigned width;

	if (op.type().is_bool())
	{
      boolbv_get_width(expr.type(), width);

      zero = boolector_int(boolector_ctx, 0, width);
	  one =  boolector_int(boolector_ctx, 1, width);

	  if (convert_bv(op, operand))
	    return true;

	  result = boolector_cond(boolector_ctx, operand, one, zero);
	}
  }
  bv = result;

#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_struct

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_struct(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}

/*******************************************************************
 Function: boolector_convt::convert_union

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_union(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}

/*******************************************************************
 Function: boolector_convt::convert_boolector_pointer

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_boolector_pointer(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}

/*******************************************************************
 Function: boolector_convt::convert_zero_string

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_zero_string(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  BtorExp *zero_string;

  create_boolector_array(expr.type(), "zero_string", zero_string);

  zero_string = bv;
  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_array

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_array(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif
}

/*******************************************************************
 Function: boolector_convt::convert_constant_array

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_constant_array(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  unsigned int width=0, i=0;
  BtorExp *array_cte, *int_cte, *val_cte;
  std::string value_cte, tmp, identifier;
  char i_str[2];

  width = atoi(expr.type().subtype().width().c_str());
  identifier = expr.identifier().as_string() + expr.type().subtype().width().c_str();

  create_boolector_array(expr.type(), identifier, array_cte);

  i=0;
  forall_operands(it, expr)
  {
	//std::cout << "i: " << i << std::endl;
	sprintf(i_str,"%i",i);
	//std::cout << "atoi(i_str): " << atoi(i_str) << std::endl;
    int_cte = boolector_int(boolector_ctx, atoi(i_str), config.ansi_c.int_width);
	if (is_signed(it->type()))
	  value_cte = integer2string(binary2integer(it->value().as_string().c_str(), true),10);
	else
	  value_cte = integer2string(binary2integer(it->value().as_string().c_str(), false),10);
	//std::cout << "value_cte.c_str(): " << value_cte.c_str() << std::endl;
	//std::cout << "width: " << width << std::endl;
	val_cte = boolector_int(boolector_ctx, atoi(value_cte.c_str()), width);
	array_cte = boolector_write(boolector_ctx, array_cte, int_cte, val_cte);
	++i;
  }

  bv = array_cte;
  return false;
}

/*******************************************************************
 Function: boolector_convt::extract_magnitude

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

std::string boolector_convt::extract_magnitude(std::string v, unsigned width)
{
  return integer2string(binary2integer(v.substr(0, width/2), true), 10);
}

/*******************************************************************
 Function: boolector_convt::extract_fraction

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

std::string boolector_convt::extract_fraction(std::string v, unsigned width)
{
  return integer2string(binary2integer(v.substr(width/2, width), false), 10);
}

/*******************************************************************
 Function: boolector_convt::convert_constant

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_constant(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "expr.pretty(): " << expr.pretty() << std::endl;
#endif

  BtorExp *const_var;
  std::string value;
  unsigned int width;

  if (expr.type().id() == "c_enum")
  {
    value = expr.value().as_string();
  }
  else if (expr.type().is_bool())
  {
    value = expr.value().as_string();
  }
  else if (expr.type().is_pointer() && expr.value().as_string() == "NULL")
  {
    value = "0";
  }
  else if (is_signed(expr.type()))
    value = integer2string(binary2integer(expr.value().as_string(), true),10);
  else
	value = integer2string(binary2integer(expr.value().as_string(), false),10);

  width = atoi(expr.type().width().c_str());

  if (expr.type().is_bool())
  {
	if (expr.is_false())
	  const_var = boolector_false(boolector_ctx);
	else if (expr.is_true())
	  const_var = boolector_true(boolector_ctx);
  }
  else if (expr.type().is_signedbv() || expr.type().id() == "c_enum")
  {
	const_var = boolector_int(boolector_ctx, atoi(value.c_str()), width);
  }
  else if (expr.type().is_unsignedbv())
  {
	const_var = boolector_unsigned_int(boolector_ctx, atoi(value.c_str()), width);
  }
  else if (expr.type().is_fixedbv())
  {
	BtorExp *magnitude, *fraction;
	std::string m, f, c;

	m = extract_magnitude(expr.value().as_string(), width);
	f = extract_fraction(expr.value().as_string(), width);

	magnitude = boolector_int(boolector_ctx, atoi(m.c_str()), width/2);
	fraction = boolector_int(boolector_ctx, atoi(f.c_str()), width/2);
	const_var = boolector_concat(boolector_ctx, magnitude, fraction);
	//const_var = boolector_int(boolector_ctx, atoi(value.c_str()), width);
  }
  else if (expr.type().is_array())
  {
	convert_constant_array(expr, const_var);
  }
  else if (expr.type().is_pointer())
  {
	width = atoi(expr.type().subtype().width().c_str());
	const_var = boolector_int(boolector_ctx, atoi(value.c_str()), width);
  }

  bv = const_var;
  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_concatenation

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_concatenation(const exprt &expr, BtorExp* &bv) {

#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}

/*******************************************************************
 Function: boolector_convt::convert_bitwise

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_bitwise(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()==2);

  BtorExp *args[2];

  if (convert_bv(expr.op0(), args[0]))
    return true;

  if (convert_bv(expr.op1(), args[1]))
	return true;

  if(expr.id()=="bitand")
    bv = boolector_and(boolector_ctx, args[0], args[1]);
  else if(expr.id()=="bitor")
	bv = boolector_or(boolector_ctx, args[0], args[1]);
  else if(expr.id()=="bitxor")
	bv = boolector_xor(boolector_ctx, args[0], args[1]);
  else if (expr.id()=="bitnand")
	bv = boolector_nand(boolector_ctx, args[0], args[1]);
  else if (expr.id()=="bitnor")
	bv = boolector_nor(boolector_ctx, args[0], args[1]);
  else if (expr.id()=="bitnxor")
    bv = boolector_xnor(boolector_ctx, args[0], args[1]);

#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_unary_minus

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_unary_minus(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "expr.op0().pretty(): " << expr.op0().pretty() << std::endl;
#endif

  assert(expr.operands().size()==1);

  BtorExp* result;

  if (convert_bv(expr.op0(), result))
    return true;

  bv = boolector_neg(boolector_ctx, result);

#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_if

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_if(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "expr.pretty(): " << expr.pretty() << std::endl;
#endif

  assert(expr.operands().size()==3);

  BtorExp *result, *operand0, *operand1, *operand2;

  if (convert_bv(expr.op0(), operand0)) return true;
  if (convert_bv(expr.op1(), operand1)) return true;
  if (convert_bv(expr.op2(), operand2)) return true;

  result = boolector_cond(boolector_ctx, operand0, operand1, operand2);

  bv = result;
  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_logical_ops

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_logical_ops(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.type().is_bool());
  assert(expr.operands().size()>=1);

  u_int i=0, size;

  size=expr.operands().size();
  BtorExp *args[size], *result;

  if (size==1)
  {
    if (convert_bv(expr.op0(), result)) return true;
  }
  else
  {
	forall_operands(it, expr)
	{
	  if (convert_bv(*it, args[i])) return true;

	  if (i>=1)
	  {
		if(expr.is_and())
		  args[i] = boolector_and(boolector_ctx, args[i-1], args[i]);
		else if(expr.id()=="or")
		  args[i] = boolector_or(boolector_ctx, args[i-1], args[i]);
		else if(expr.id()=="xor")
		  args[i] = boolector_xor(boolector_ctx, args[i-1], args[i]);
	  }

	  ++i;
	}

	result = args[size-1];
  }

  bv = result;
  return false;
}


/*******************************************************************
 Function: boolector_convt::convert_logical_not

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_logical_not(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "expr.pretty(): " << expr.pretty() << std::endl;
#endif

  assert(expr.operands().size()==1);

  BtorExp *operand0;

  if (convert_bv(expr.op0(), operand0)) return true;

  bv = boolector_not(boolector_ctx, operand0);
  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_equality

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_equality(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()==2);
  assert(expr.op0().type()==expr.op1().type());

  BtorExp *result=0, *args[2];

  if (convert_bv(expr.op0(), args[0])) return true;
  if (convert_bv(expr.op1(), args[1])) return true;

  if (expr.id()=="=")
    result = boolector_eq(boolector_ctx, args[0], args[1]);
  else
	result = boolector_ne(boolector_ctx, args[0], args[1]);

  bv = result;
  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_add

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_add(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "expr.pretty(): " << expr.pretty() << std::endl;
#endif

  assert(expr.operands().size()>=2);
  u_int i=0, size;

  size=expr.operands().size()+1;
  BtorExp *args[size];

  forall_operands(it, expr)
  {
	if (convert_bv(*it, args[i])) return true;
    if (i==1)
      args[size-1] = boolector_add(boolector_ctx, args[0], args[1]);
    else if (i>1)
 	  args[size-1] = boolector_add(boolector_ctx, args[size-1], args[i]);
    ++i;
  }
  bv = args[i];
  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_sub

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_sub(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()>=2);
  u_int i=0, size;

  size=expr.operands().size()+1;
  BtorExp *args[size];

  forall_operands(it, expr)
  {
	if (convert_bv(*it, args[i])) return true;

    if (i==1)
      args[size-1] = boolector_sub(boolector_ctx, args[0], args[1]);
    else if (i>1)
 	  args[size-1] = boolector_sub(boolector_ctx, args[size-1], args[i]);
    ++i;
  }

  bv = args[i];
  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_div

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_div(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()==2);

  BtorExp *result, *args[2], *concat;

  if (convert_bv(expr.op0(), args[0])) return true;
  if (convert_bv(expr.op1(), args[1])) return true;

  if (expr.type().is_signedbv())
	result = boolector_sdiv(boolector_ctx, args[0], args[1]);
  else if (expr.type().is_unsignedbv())
	result = boolector_udiv(boolector_ctx, args[0], args[1]);
  else if (expr.type().is_fixedbv())
  {
    fixedbvt fbt(expr);
    unsigned fraction_bits=fbt.spec.get_fraction_bits();

    concat = boolector_concat(boolector_ctx, args[0], boolector_int(boolector_ctx, 0, fraction_bits));
    result = boolector_sdiv(boolector_ctx, concat, boolector_sext(boolector_ctx, args[1], fraction_bits));
    result = boolector_slice(boolector_ctx, result, fbt.spec.width-1, 0);
  }
  else
    throw "unsupported type for /: "+expr.type().id_string();

  bv = result;
  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_mod

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_mod(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()==2);

  BtorExp *result, *operand0, *operand1;

  if (convert_bv(expr.op0(), operand0)) return true;
  if (convert_bv(expr.op1(), operand1)) return true;

  if(expr.type().is_signedbv())
	result = boolector_srem(boolector_ctx, operand0, operand1);
  else if (expr.type().is_unsignedbv())
	result = boolector_urem(boolector_ctx, operand0, operand1);
  else if (expr.type().is_fixedbv())
	throw "unsupported type for mod: "+expr.type().id_string();

  bv = result;
  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_mul

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_mul(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()>=2);
  u_int i=0, size;
  unsigned fraction_bits;
  size=expr.operands().size()+1;
  BtorExp *args[size];

  if(expr.type().is_fixedbv())
  {
    fixedbvt fbt(expr);
	fraction_bits=fbt.spec.get_fraction_bits();
  }

  forall_operands(it, expr)
  {
	if (convert_bv(*it, args[i])) return true;

    if(expr.type().is_fixedbv())
      args[i] = boolector_sext(boolector_ctx, args[i], fraction_bits);

    if (i==1)
      args[size-1] = boolector_mul(boolector_ctx, args[0], args[1]);
    else if (i>1)
 	  args[size-1] = boolector_mul(boolector_ctx, args[size-1], args[i]);
    ++i;
  }

  if(expr.type().is_fixedbv())
  {
	fixedbvt fbt(expr);
    args[i] = boolector_slice(boolector_ctx, args[i], fbt.spec.width+fraction_bits-1, fraction_bits);
  }

  bv = args[i];
  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_pointer

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_pointer(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()==1);
  assert(expr.type().is_pointer());

  BtorExp *result, *args[2];
  std::string symbol_name;

  if (expr.op0().is_index())
  {
    const exprt &object=expr.op0().operands()[0];
	const exprt &index=expr.op0().operands()[1];

    read_cache(object, args[0]);
	if (convert_bv(index, args[1])) return true;

	result = boolector_read(boolector_ctx, args[0], args[1]);
  }

  bv = result;
  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_array_of

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_array_of(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  BtorExp *index, *value, *array_of_var;
  const array_typet &array_type_size=to_array_type(expr.type());
  std::string tmp;
  unsigned int j, size, width;

  tmp = integer2string(binary2integer(array_type_size.size().value().as_string(), false),10);
  size = atoi(tmp.c_str());
  width = atoi(expr.type().subtype().width().c_str());

  if (expr.type().subtype().is_bool())
  {
    value = boolector_false(boolector_ctx);
    array_of_var = boolector_array(boolector_ctx, 1, config.ansi_c.int_width,"ARRAY_OF(false)");
  }
  else if (expr.type().subtype().is_signedbv() || expr.type().subtype().is_unsignedbv())
  {
	if (convert_bv(expr.op0(), value)) return true;
    array_of_var = boolector_array(boolector_ctx, width, config.ansi_c.int_width,"ARRAY_OF(0)");
  }
  else if (expr.type().subtype().is_fixedbv())
  {
	if (convert_bv(expr.op0(), value)) return true;
    array_of_var = boolector_array(boolector_ctx, width, config.ansi_c.int_width, "ARRAY_OF(0l)");
  }
  else if (expr.type().subtype().is_pointer())
  {
	const exprt &object=expr.op0().operands()[0];
	const exprt &index=expr.op0().operands()[1];

	width = atoi(expr.op0().type().subtype().width().c_str());
	if (convert_bv(expr.op0(), value)) return true;
	array_of_var = boolector_array(boolector_ctx, width, config.ansi_c.int_width, "&(ZERO_STRING())[0]");
  }

  if (size==0)
	size=1; //update at leat the first element of the array of bool

  //update array
  for (j=0; j<size; j++)
  {
    index = boolector_int(boolector_ctx, j, config.ansi_c.int_width);
    array_of_var = boolector_write(boolector_ctx, array_of_var, index, value);
  }

  bv = array_of_var;
  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_array_of_array

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_array_of_array(const std::string identifier, const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}


/*******************************************************************
 Function: boolector_convt::convert_index

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_index(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "expr.op0().pretty(): " << expr.op0().pretty() << std::endl;
  std::cout << "expr.op1().pretty(): " << expr.op1().pretty() << std::endl;
#endif

  assert(expr.operands().size()==2);

  BtorExp *array, *index;

  if (convert_bv(expr.op0(), array)) return true;
  if (convert_bv(expr.op1(), index)) return true;
#if 0
  if (expr.op0().is_constant() && expr.op1().is_symbol())
  {
    const array_typet &array_type_size=to_array_type(expr.op0().type());
    std::string tmp;
    unsigned size;
    BtorExp *lower, *upper, *formula;

    tmp = integer2string(binary2integer(array_type_size.size().value().as_string(), false),10);
    size = atoi(tmp.c_str());
    std::cout << "size: " << size << std::endl;
    std::cout << "expr.op1().type().id(): " << expr.op1().type().id() << std::endl;
    if (expr.op1().type().is_signedbv())
    {
      unsigned width = atoi(expr.op1().type().width().c_str());
      std::cout << "width: " << width << std::endl;
      lower = boolector_sgte(boolector_ctx, index, boolector_zero (boolector_ctx, width));
      upper = boolector_slt(boolector_ctx, index, boolector_int(boolector_ctx, size, width));
      formula = boolector_and(boolector_ctx, lower, upper);
      boolector_assert(boolector_ctx, formula);
      index = boolector_zero (boolector_ctx, width);
    }
  }
  std::cout << "boolector_is_array(boolector_ctx, array): " << boolector_is_array(boolector_ctx, array) << std::endl;
  std::cout << "boolector_is_array(boolector_ctx, index): " << boolector_is_array(boolector_ctx, index) << std::endl;
  std::cout << "boolector_bv_assignment: " << boolector_bv_assignment(boolector_ctx, boolector_read(boolector_ctx, array, index)) << std::endl;
#endif
  bv = boolector_read(boolector_ctx, array, index);
  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_constant

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_shift_constant(const exprt &expr, unsigned int wop0, unsigned int wop1, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "expr.pretty(): " << expr.pretty() << std::endl;
#endif

  BtorExp *result;
  std::string value;
  unsigned int size, width;

  if (is_signed(expr.type()))
    value = integer2string(binary2integer(expr.value().as_string(), true),10);
  else
	value = integer2string(binary2integer(expr.value().as_string(), false),10);

  size = atoi(expr.type().width().c_str());

  if (wop0>wop1)
    width = (log(wop0)/log(2))+1;
  else
    width = log(size)/log(2);


  if (expr.type().is_signedbv() || expr.type().id() == "c_enum")
	result = boolector_int(boolector_ctx, atoi(value.c_str()), width);
  else if (expr.type().is_unsignedbv())
	result = boolector_unsigned_int(boolector_ctx, atoi(value.c_str()), width);
  else if (expr.type().is_fixedbv())
	result = boolector_int(boolector_ctx, atoi(value.c_str()), width);

  bv = result;
  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_shift

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_shift(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()==2);

  BtorExp *result, *operand0, *operand1;

  unsigned width_op0, width_op1;
  boolbv_get_width(expr.op0().type(), width_op0);
  boolbv_get_width(expr.op1().type(), width_op1);

  if (expr.op0().is_constant())
	convert_shift_constant(expr.op0(), width_op0, width_op1, operand0);
  else
    if (convert_bv(expr.op0(), operand0)) return true;

  if (expr.op1().is_constant())
	convert_shift_constant(expr.op1(), width_op0, width_op1, operand1);
  else
    if (convert_bv(expr.op1(), operand1)) return true;

  if(expr.is_ashr())
    result = boolector_sra(boolector_ctx, operand0, operand1);
  else if (expr.id()=="lshr")
    result = boolector_srl(boolector_ctx, operand0, operand1);
  else if(expr.id()=="shl")
    result = boolector_sll(boolector_ctx, operand0, operand1);
  else
    assert(false);

  bv=result;
  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_with

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_with(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()>=1);
  BtorExp *result, *array, *index, *value;

  if (convert_bv(expr.op0(), array)) return true;
  if (convert_bv(expr.op1(), index)) return true;
  if (convert_bv(expr.op2(), value)) return true;

  result = boolector_write(boolector_ctx, array, index, value);

  bv = result;

  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_abs

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_abs(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  unsigned width;
  std::string out;

  boolbv_get_width(expr.type(), width);

  const exprt::operandst &operands=expr.operands();

  if(operands.size()!=1)
    throw "abs takes one operand";

  const exprt &op0=expr.op0();
  static BtorExp *zero, *minus_one, *is_negative, *val_orig, *val_mul;

  out = "width: "+ width;
  if (expr.type().is_signedbv() || expr.type().is_fixedbv())
    zero = boolector_int(boolector_ctx, 0, width);
  else if (expr.type().is_unsignedbv())
	zero = boolector_unsigned_int(boolector_ctx, 0, width);

  minus_one = boolector_int(boolector_ctx, -1, width);

  if (convert_bv(op0, val_orig)) return true;

  if (expr.type().is_signedbv() || expr.type().is_fixedbv())
    is_negative = boolector_slt(boolector_ctx, val_orig, zero);

  val_mul = boolector_mul(boolector_ctx, val_orig, minus_one);

  bv = boolector_cond(boolector_ctx, is_negative, val_mul, val_orig);

  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_bitnot

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_bitnot(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}

/*******************************************************************
 Function: boolector_convt::convert_bitnot

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

unsigned int boolector_convt::convert_member_name(const exprt &lhs, const exprt &rhs)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}


/*******************************************************************
 Function: boolector_convt::convert_extractbit

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_extractbit(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif
}

/*******************************************************************
 Function: boolector_convt::convert_object

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_object(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}

/*******************************************************************
 Function: boolector_convt::select_pointer_offset

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::select_pointer_offset(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  assert(expr.operands().size()==1);

  BtorExp *offset;

  if (convert_bv(expr.op0(), offset)) return true;

  bv = offset;
  return false;
}

/*******************************************************************
 Function: boolector_convt::convert_member

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_member(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}


/*******************************************************************
 Function: convert_invalid_pointer

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_invalid_pointer(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}

/*******************************************************************
 Function: convert_pointer_object

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_pointer_object(const exprt &expr, BtorExp* &bv)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

}
/*******************************************************************
 Function: boolector_convt::convert_boolector_expr

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::convert_boolector_expr(const exprt &expr, BtorExp* &bv)
{
  if (expr.is_symbol())
	return convert_identifier(expr.identifier().as_string(), expr.type(), bv);
  else if (expr.id() == "nondet_symbol") {
	return convert_identifier("nondet$"+expr.identifier().as_string(), expr.type(), bv);
  } else if (expr.is_typecast())
    return convert_typecast(expr, bv);
#if 0
  else if (expr.is_struct())
	return convert_struct(expr);
  else if (expr.is_union())
	return convert_union(expr);
#endif
  else if (expr.is_constant())
	return convert_constant(expr, bv);
  else if (expr.id() == "concatenation")
	return convert_concatenation(expr, bv);
  else if (expr.id() == "bitand" || expr.id() == "bitor" || expr.id() == "bitxor"
		|| expr.id() == "bitnand" || expr.id() == "bitnor" || expr.id() == "bitnxor")
    return convert_bitwise(expr, bv);
  else if (expr.id() == "bitnot")
	return convert_bitnot(expr, bv);
  else if (expr.id() == "unary-")
    return convert_unary_minus(expr, bv);
  else if (expr.id() == "if")
    return convert_if(expr, bv);
  else if (expr.is_and() || expr.id() == "or" || expr.id() == "xor")
	return convert_logical_ops(expr, bv);
  else if (expr.is_not())
	return convert_logical_not(expr, bv);
  else if (expr.id() == "=" || expr.is_notequal())
	return convert_equality(expr, bv);
  else if (expr.id() == "<=" || expr.id() == "<" || expr.id() == ">="
		|| expr.id() == ">")
	return convert_rel(expr, bv);
  else if (expr.id() == "+")
	return convert_add(expr, bv);
  else if (expr.id() == "-")
	return convert_sub(expr, bv);
  else if (expr.id() == "/")
	return convert_div(expr, bv);
  else if (expr.id() == "mod")
	return convert_mod(expr, bv);
  else if (expr.id() == "*")
	return convert_mul(expr, bv);
  else if(expr.id()=="abs")
    return convert_abs(expr, bv);
  else if (expr.is_address_of() || expr.id() == "implicit_address_of"
		|| expr.id() == "reference_to")
	return convert_pointer(expr, bv);
  else if (expr.is_array_of())
	return convert_array_of(expr, bv);
  else if (expr.is_index())
	return convert_index(expr, bv);
  else if (expr.is_ashr() || expr.id() == "lshr" || expr.id() == "shl")
	return convert_shift(expr, bv);
  else if (expr.id() == "with")
	return convert_with(expr, bv);
  else if (expr.is_member())
	return convert_member(expr, bv);
#if 0
  else if (expr.id() == "invalid-pointer")
	return convert_invalid_pointer(expr);
#endif
  else if (expr.id()=="zero_string")
	return convert_zero_string(expr, bv);
  else if (expr.id() == "pointer_offset")
	return select_pointer_offset(expr, bv);
  else if (expr.id() == "pointer_object")
	return convert_pointer_object(expr, bv);
#if 0
  else if (expr.id() == "same-object")
	return convert_object(expr);
#endif
  else if (expr.id() == "string-constant") {
	  exprt tmp;
	  string2array(expr, tmp);
	return convert_boolector_expr(tmp, bv);
  } else if (expr.id() == "extractbit")
	return convert_extractbit(expr, bv);
  else if (expr.id() == "replication") {
	assert(expr.operands().size()==2);
  } else
	throw "convert_boolector_expr: " + expr.id_string() + " is not supported yet";
}

/*******************************************************************
 Function: boolector_convt::assign_boolector_expr

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool boolector_convt::assign_boolector_expr(const exprt expr)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
#endif

  if (expr.op0().type().is_pointer() && expr.op0().type().subtype().is_code())
  {
	ignoring(expr);
	return false;
  }
  else if (expr.op0().type().is_array() && expr.op0().type().subtype().is_struct())
  {
	ignoring(expr);
  	return false;
  }
  else if (expr.op0().type().is_pointer() && expr.op0().type().subtype().is_symbol())
  {
	ignoring(expr);
  	return false;
  }

  return true;
}


/*******************************************************************
 Function: boolector_convt::set_to

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void boolector_convt::set_to(const exprt &expr, bool value)
{
#ifdef DEBUG
  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
  std::cout << "set_to expr.pretty(): " << expr.pretty() << std::endl;

#endif

#ifdef DEBUG
  std::cout << "1 - set_to expr.pretty(): " << expr.pretty() << std::endl;
  std::cout << "2 - set_to expr.id(): " << expr.id() << std::endl;
  std::cout << "3 - expr.type().id(): " << expr.type().id() << std::endl;
  std::cout << "4 - value: " << value << std::endl;
#endif


  if(expr.is_and() && value)
  {
    forall_operands(it, expr)
      set_to(*it, true);
    return;
  }

  if(expr.is_not())
  {
    assert(expr.operands().size()==1);
    return set_to(expr.op0(), !value);
  }

#if 1
  if(!expr.type().is_bool())
  {
    std::string msg="prop_convt::set_to got "
                    "non-boolean expression:\n";
    msg+=expr.to_string();
    throw msg;
  }

  bool boolean=true;

  forall_operands(it, expr)
    if(!it->type().is_bool())
    {
      boolean=false;
      break;
    }

#ifdef DEBUG
  std::cout << "boolean: " << boolean << std::endl;
#endif

  if(boolean)
  {
    if(expr.is_not())
    {
      if(expr.operands().size()==1)
      {
        set_to(expr.op0(), !value);
        return;
      }
    }
    else
    {
      if(value)
      {
        // set_to_true
        if(expr.is_and())
        {
          forall_operands(it, expr)
            set_to_true(*it);
          return;
        }
        else if(expr.id()=="or")
        {
          if(expr.operands().size()>0)
          {
            bvt bv;
            bv.reserve(expr.operands().size());

            forall_operands(it, expr)
              bv.push_back(convert(*it));
            prop.lcnf(bv);
            return;
          }
        }
        else if(expr.id()=="=>")
        {
          if(expr.operands().size()==2)
          {
            bvt bv;
            bv.resize(2);
            bv[0]=prop.lnot(convert(expr.op0()));
            bv[1]=convert(expr.op1());
            prop.lcnf(bv);
            return;
          }
        }
#if 0
        else if(expr.id()=="=")
        {
          if(!set_equality_to_true(expr))
            return;
        }
#endif
      }
      else
      {
        // set_to_false
        if(expr.id()=="=>") // !(a=>b)  ==  (a && !b)
        {
          if(expr.operands().size()==2)
          {
            set_to_true(expr.op0());
            set_to_false(expr.op1());
          }
        }
        else if(expr.id()=="or") // !(a || b)  ==  (!a && !b)
        {
          forall_operands(it, expr)
            set_to_false(*it);
        }
      }
    }
  }

  // fall back to convert
  prop.l_set_to(convert(expr), value);

  if (value && expr.is_and())
  {
	forall_operands(it, expr)
	  set_to(*it, true);
	return;
  }

  if (value && expr.is_true())
	return;
#endif

  if (expr.id() == "=" && value)
  {
    assert(expr.operands().size()==2);

    BtorExp *formula, *operand0, *operand1;

	if (assign_boolector_expr(expr))
	{
#ifdef DEBUG
	  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
	  std::cout << "set_to expr.op0().pretty():" << expr.op0().pretty() << std::endl;
#endif
	  if (convert_bv(expr.op0(), operand0))
	  {
		assert(0);
		return ;
	  }
#ifdef DEBUG
	  std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
	  std::cout << "set_to expr.op1().pretty():" << expr.op1().pretty() << std::endl;
#endif
	  if (convert_bv(expr.op1(), operand1))
	  {
		assert(0);
		return ;
	  }

	  if (expr.op0().type().is_bool())
	    formula = boolector_iff(boolector_ctx, operand0, operand1);
	  else
	    formula = boolector_eq(boolector_ctx, operand0, operand1);

	  //formula = boolector_eq(boolector_ctx, operand0, operand1);
	  boolector_assert(boolector_ctx, formula);

	  if (boolector_prop.btor)
	    boolector_prop.assumpt.push_back(formula);

	  //boolector_dump_btor(boolector_ctx, btorFile, formula);
	  //boolector_dump_smt(boolector_ctx, smtFile, formula);
	}
  }

}

/*******************************************************************
 Function: boolector_convt::get_number_variables_z

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

unsigned int boolector_convt::get_number_variables_boolector(void)
{
  return number_variables_boolector;
}

