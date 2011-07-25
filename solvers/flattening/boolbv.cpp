/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <map>
#include <set>

#include <symbol.h>
#include <mp_arith.h>
#include <i2string.h>
#include <arith_tools.h>
#include <replace_expr.h>
#include <string2array.h>
#include <std_types.h>
#include <prefix.h>
#include <std_expr.h>

#include "boolbv.h"
#include "boolbv_type.h"

#ifdef HAVE_FLOATBV
#include "../floatbv/float_utils.h"
#endif

//#define DEBUG

/*******************************************************************\

Function: boolbvt::literal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool boolbvt::literal(
  const exprt &expr,
  const unsigned bit,
  literalt &dest) const
{
  if(expr.type().id()=="bool")
  {
    assert(bit==0);
    return prop_convt::literal(expr, dest);
  }
  else
  {
    if(expr.id()=="symbol" ||
       expr.id()=="nondet_symbol")
    {
      const irep_idt &identifier=expr.get("identifier");

      boolbv_mapt::mappingt::const_iterator it_m=
        map.mapping.find(identifier);

      if(it_m==map.mapping.end()) return true;

      const boolbv_mapt::map_entryt &map_entry=it_m->second;
      
      assert(bit<map_entry.literal_map.size());
      if(!map_entry.literal_map[bit].is_set) return true;

      dest=map_entry.literal_map[bit].l;
      return false;
    }
    else if(expr.id()=="index")
    {
      if(expr.operands().size()!=2)
        throw "index takes two operands";

      unsigned element_width;

      if(boolbv_get_width(expr.type(), element_width))
        throw "literal expects a bit-vector type";

      mp_integer index;
      if(to_integer(expr.op1(), index))
        throw "literal expects constant index";

      unsigned offset=integer2long(index*element_width);

      return literal(expr.op0(), bit+offset, dest);
    }
    else if(expr.id()=="member")
    {
      if(expr.operands().size()!=1)
        throw "member takes one operand";

      const irept &components=expr.type().find("components");
      const irep_idt &component_name=expr.get("component_name");

      unsigned offset=0;

      forall_irep(it, components.get_sub())
      {
        const typet &subtype=it->type();

        if(it->get("name")==component_name)
          return literal(expr.op0(), bit+offset, dest);

        unsigned element_width;
        if(boolbv_get_width(subtype, element_width))
          throw "literal expects a bit-vector type";

        offset+=element_width;
      }

      throw "failed to find component";
    }
  }

  throw "found no literal for expression";
}

/*******************************************************************\

Function: boolbvt::convert_bv

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::convert_bv(const exprt &expr, bvt &bv)
{
  // check cache first
  
  {
    bv_cachet::const_iterator cache_result=bv_cache.find(expr);
    if(cache_result!=bv_cache.end())
    {
      //std::cerr << "Cache hit on " << expr << "\n";
      bv=cache_result->second;
      return;
    }
  }
  
  convert_bitvector(expr, bv);

  // insert into cache
  bv_cache.insert(std::pair<const exprt, bvt>(expr, bv));
}

/*******************************************************************\

Function: boolbvt::conversion_failed

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::conversion_failed(const exprt &expr, bvt &bv)
{
  ignoring(expr);

  // try to make it free bits
  unsigned width;
  if(boolbv_get_width(expr.type(), width))
    bv.clear();
  else
  {
    bv.resize(width);

    for(unsigned i=0; i<width; i++)
      bv[i]=prop.new_variable();
  }
}

/*******************************************************************\

Function: boolbvt::convert_bitvector

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::convert_bitvector(const exprt &expr, bvt &bv)
{
  #ifdef DEBUG
  std::cout << "BV: " << expr.pretty() << std::endl;
  #endif

  if(expr.type().id()=="bool")
  {
    bv.resize(1);
    bv[0]=convert(expr);
    return;
  }

  if(expr.id()=="index")
    return convert_index(to_index_expr(expr), bv);
  else if(expr.id()=="constraint_select_one")
    return convert_constraint_select_one(expr, bv);
  else if(expr.id()=="member")
    return convert_member(expr, bv);
  else if(expr.id()=="with")
    return convert_with(expr, bv);
  else if(expr.id()=="width")
  {
    unsigned result_width;
    if(boolbv_get_width(expr.type(), result_width))
      return conversion_failed(expr, bv);

    if(expr.operands().size()!=1)
      return conversion_failed(expr, bv);

    unsigned op_width;
    if(boolbv_get_width(expr.op0().type(), op_width))
      return conversion_failed(expr, bv);

    bv.resize(result_width);

    if(expr.type().id()=="unsignedbv" ||
       expr.type().id()=="signedbv")
    {
      std::string binary=integer2binary(op_width/8, result_width);

      for(unsigned i=0; i<result_width; i++)
      {
        bool bit=(binary[binary.size()-i-1]=='1');
        bv[i]=const_literal(bit);
      }

      return;
    }
  }
  else if(expr.id()=="case")
    return convert_case(expr, bv);
  else if(expr.id()=="cond")
    return convert_cond(expr, bv);
  else if(expr.id()=="if")
    return convert_if(expr, bv);
  else if(expr.id()=="constant")
    return convert_constant(expr, bv);
  else if(expr.id()=="typecast")
    return convert_typecast(expr, bv);
  else if(expr.id()=="symbol")
    return convert_symbol(expr, bv);
  else if(expr.id()=="bv_literals")
    return convert_bv_literals(expr, bv);
  else if(expr.id()=="+" || expr.id()=="-" ||
          expr.id()=="no-overflow-plus" ||
          expr.id()=="no-overflow-minus")
    return convert_add_sub(expr, bv);
  else if(expr.id()=="*" ||
          expr.id()=="no-overflow-mult")
    return convert_mult(expr, bv);
  else if(expr.id()=="/")
    return convert_div(expr, bv);
  else if(expr.id()=="mod")
    return convert_mod(expr, bv);
  else if(expr.id()=="shl" || expr.id()=="ashr" || expr.id()=="lshr")
    return convert_shift(expr, bv);
  else if(expr.id()=="concatenation")
    return convert_concatenation(expr, bv);
  else if(expr.id()=="replication")
    return convert_replication(expr, bv);
  else if(expr.id()=="extractbits")
    return convert_extractbits(expr, bv);
  else if(expr.id()=="bitnot" || expr.id()=="bitand" ||
          expr.id()=="bitor" || expr.id()=="bitxor")
    return convert_bitwise(expr, bv);
  else if(expr.id()=="unary-" ||
          expr.id()=="no-overflow-unary-minus")
    return convert_unary_minus(expr, bv);
  else if(expr.id()=="abs")
    return convert_abs(expr, bv);
  else if(has_prefix(expr.id_string(), "byte_extract"))
    return convert_byte_extract(expr, bv);
  else if(has_prefix(expr.id_string(), "byte_update"))
    return convert_byte_update(expr, bv);
  else if(expr.id()=="nondet_symbol" ||
          expr.id()=="quant_symbol")
    return convert_symbol(expr, bv);
  else if(expr.id()=="struct")
    return convert_struct(expr, bv);
  else if(expr.id()=="union")
    return convert_union(expr, bv);
  else if(expr.id()=="string-constant")
  {
    exprt tmp;
    string2array(expr, tmp);
    return convert_bitvector(tmp, bv);
  }
  else if(expr.id()=="array")
    return convert_array(expr, bv);
  else if(expr.id()=="lambda")
    return convert_lambda(expr, bv);
  else if(expr.id()=="array_of")
    return convert_array_of(expr, bv);

  return conversion_failed(expr, bv);
}

/*******************************************************************\

Function: boolbvt::convert_array

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::convert_array(const exprt &expr, bvt &bv)
{
  unsigned width;
  if(boolbv_get_width(expr.type(), width))
    return conversion_failed(expr, bv);
    
  bv.resize(width);

  if(expr.type().id()=="array")
  {
    unsigned op_width=width/expr.operands().size();
    unsigned offset=0;
    
    forall_operands(it, expr)
    {
      bvt tmp;

      convert_bv(*it, tmp);

      if(tmp.size()!=op_width)
        throw "convert_array: unexpected operand width";

      for(unsigned j=0; j<op_width; j++)
        bv[offset+j]=tmp[j];

      offset+=op_width;
    }   

    return;
  }
  
  conversion_failed(expr, bv);
}

/*******************************************************************\

Function: boolbvt::convert_lambda

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::convert_lambda(const exprt &expr, bvt &bv)
{
  unsigned width;
  if(boolbv_get_width(expr.type(), width))
    return conversion_failed(expr, bv);

  if(expr.operands().size()!=2)
    throw "lambda takes two operands";

  if(expr.type().id()!="array")
    return conversion_failed(expr, bv);

  const exprt &array_size=
    to_array_type(expr.type()).size();

  mp_integer size;

  if(to_integer(array_size, size))
    return conversion_failed(expr, bv);

  typet counter_type=expr.op0().type();

  for(mp_integer i=0; i<size; ++i)
  {
    exprt counter=from_integer(i, counter_type);

    exprt expr(expr.op1());
    replace_expr(expr.op0(), counter, expr);

    bvt tmp;
    convert_bv(expr, tmp);

    unsigned offset=integer2long(i*tmp.size());

    if(size*tmp.size()!=width)
      throw "convert_lambda: unexpected operand width";

    for(unsigned j=0; j<tmp.size(); j++)
      bv[offset+j]=tmp[j];
  }
}

/*******************************************************************\

Function: boolbvt::convert_bv_literals

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::convert_bv_literals(const exprt &expr, bvt &bv)
{
  unsigned width;
  if(boolbv_get_width(expr.type(), width))
    return conversion_failed(expr, bv);

  bv.resize(width);

  const irept::subt &bv_sub=expr.find("bv").get_sub();

  if(bv_sub.size()!=width)
    throw "bv_literals with wrong size";

  for(unsigned i=0; i<width; i++)
    bv[i].set(atol(bv_sub[i].id().c_str()));
}

/*******************************************************************\

Function: boolbvt::convert_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::convert_symbol(const exprt &expr, bvt &bv)
{
  unsigned width;
  if(boolbv_get_width(expr.type(), width))
    return conversion_failed(expr, bv);

  bv.resize(width);
  
  const irep_idt &identifier=expr.get("identifier");

  if(identifier.empty())
    throw "got empty identifier";

  for(unsigned i=0; i<width; i++)
    bv[i]=map.get_literal(identifier, i, expr.type());

  for(unsigned i=0; i<width; i++)
    if(bv[i].var_no()>=prop.no_variables() &&
      !bv[i].is_constant()) { std::cout << identifier << std::endl; abort(); }
}
   
/*******************************************************************\

Function: boolbvt::convert_struct

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::convert_struct(const exprt &expr, bvt &bv)
{
  if(expr.type().id()!="struct")
    return conversion_failed(expr, bv);

  unsigned width;
  if(boolbv_get_width(expr.type(), width))
    return conversion_failed(expr, bv);

  const irept &components=expr.type().find("components");

  if(expr.operands().size()!=components.get_sub().size())
    throw "struct: wrong number of arguments";

  bv.resize(width);

  unsigned offset=0, i=0;
  
  forall_irep(it, components.get_sub())
  {
    const typet &subtype=it->type();
    const exprt &op=expr.operands()[i];

    if(subtype!=op.type())
      throw "struct: component type does not match: "+
        subtype.to_string()+" vs. "+
        op.type().to_string();
        
    unsigned subtype_width;
    assert(!boolbv_get_width(subtype, subtype_width));

    if(subtype_width!=0)
    {
      bvt op_bv;
      
      convert_bv(op, op_bv);
    
      assert(offset<width);
      assert(op_bv.size()==subtype_width);
      assert(offset+op_bv.size()<=width);

      for(unsigned j=0; j<op_bv.size(); j++)
        bv[offset+j]=op_bv[j];

      offset+=op_bv.size();
    }

    i++;    
  }
  
  assert(offset==width);
}

/*******************************************************************\

Function: boolbvt::convert_constraint_select_one

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::convert_constraint_select_one(const exprt &expr, bvt &bv)
{
  const exprt::operandst &operands=expr.operands();

  if(expr.id()!="constraint_select_one")
    throw "expected constraint_select_one expression";

  if(operands.size()<2)
    throw "constraint_select_one takes at least two operands";

  if(expr.type()!=expr.op0().type())
    throw "constraint_select_one expects matching types";
    
  if(prop.has_set_to())
  {
    std::vector<bvt> op_bv;
    op_bv.resize(expr.operands().size());

    unsigned i=0;
    forall_operands(it, expr)
      convert_bv(*it, op_bv[i++]);

    bv=op_bv[0];

    // add constraints

    bvt equal_bv;
    equal_bv.resize(bv.size());

    bvt b;
    b.reserve(op_bv.size()-1);

    for(unsigned i=1; i<op_bv.size(); i++)
    {
      if(op_bv[i].size()!=bv.size())
        throw "constraint_select_one expects matching width";

      for(unsigned j=0; j<bv.size(); j++)
        equal_bv[j]=prop.lequal(bv[j], op_bv[i][j]);

      b.push_back(prop.land(equal_bv));
    }

    prop.l_set_to_true(prop.lor(b));
  }
  else
  {
    unsigned op_nr=0;
    forall_operands(it, expr)
    {
      bvt op_bv;
      convert_bv(*it, op_bv);

      if(op_nr==0)
        bv=op_bv;
      else
      {
        if(op_bv.size()!=bv.size())
          return conversion_failed(expr, bv);

        for(unsigned i=0; i<op_bv.size(); i++)
          bv[i]=prop.lselect(prop.new_variable(), bv[i], op_bv[i]);
      }

      op_nr++;
    }
  }
}

/*******************************************************************\

Function: boolbvt::convert_rest

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt boolbvt::convert_rest(const exprt &expr)
{
  if(expr.type().id()!="bool")
  {
    std::cerr << expr << std::endl;
    throw "boolbvt::convert_rest got non-boolean operand";
  }

  const exprt::operandst &operands=expr.operands();

  if(expr.id()=="typecast")
    return convert_typecast(expr);
  else if(expr.id()=="=")
    return convert_equality(to_equality_expr(expr));
  else if(expr.id()=="notequal")
  {
    if(expr.operands().size()!=2)
      throw "notequal expects two operands";

    equality_exprt e(expr.op0(), expr.op1());
    return prop.lnot(convert_equality(e));
  }
  else if(expr.id()=="ieee_float_equal" ||
          expr.id()=="ieee_float_notequal")
    return convert_ieee_float_rel(expr);
  else if(expr.id()=="<=" || expr.id()==">=" ||
          expr.id()=="<"  || expr.id()==">")
    return convert_bv_rel(expr);
  else if(expr.id()=="extractbit")
    return convert_extractbit(expr);
  else if(expr.id()=="forall")
    return convert_quantifier(expr);
  else if(expr.id()=="exists")
    return convert_quantifier(expr);
  else if(expr.id()=="index")
  {
    bvt bv;
    convert_index(to_index_expr(expr), bv);
    
    if(bv.size()!=1)
      throw "convert_index returned non-bool bitvector";

    return bv[0];
  }
  else if(expr.id()=="member")
  {
    bvt bv;
    convert_member(expr, bv);
    
    if(bv.size()!=1)
      throw "convert_member returned non-bool bitvector";

    return bv[0];
  }
  else if(expr.id()=="case")
  {
    bvt bv;
    convert_case(expr, bv);

    if(bv.size()!=1)
      throw "convert_case returned non-bool bitvector";

    return bv[0];
  }
  else if(expr.id()=="cond")
  {
    bvt bv;
    convert_cond(expr, bv);

    if(bv.size()!=1)
      throw "convert_cond returned non-bool bitvector";

    return bv[0];
  }
  else if(expr.id()=="sign")
  {
    if(expr.operands().size()!=1)
      throw "sign expects one operand";

    bvt bv;
    convert_bv(operands[0], bv);

    if(bv.size()<1)
      throw "sign operator takes one non-empty operand";

    if(operands[0].type().id()=="signedbv")
      return bv[bv.size()-1];
    else if(operands[0].type().id()=="unsignedbv")
      return const_literal(false);
    else if(operands[0].type().id()=="fixedbv")
      return bv[bv.size()-1];
    else if(operands[0].type().id()=="floatbv")
      return bv[bv.size()-1];
  }
  else if(expr.id()=="reduction_or"  || expr.id()=="reduction_and"  ||
          expr.id()=="reduction_nor" || expr.id()=="reduction_nand" ||
          expr.id()=="reduction_xor" || expr.id()=="reduction_xnor")
    return convert_reduction(expr);
  else if(has_prefix(expr.id_string(), "overflow-"))
    return convert_overflow(expr);
  else if(expr.id()=="isnan")
  {
    if(expr.operands().size()!=1)
      throw "isnan expects one operand";

    bvt bv;
    convert_bv(operands[0], bv);
    
    if(expr.op0().type().id()=="floatbv")
    {
      #ifdef HAVE_FLOATBV
      float_utilst float_utils(prop);
      float_utils.spec=to_floatbv_type(expr.op0().type());
      return float_utils.is_NaN(bv);
      #endif
    }
  }
  else if(expr.id()=="isfinite")
  {
    if(expr.operands().size()!=1)
      throw "isfinite expects one operand";

    bvt bv;
    convert_bv(operands[0], bv);
    
    if(expr.op0().type().id()=="floatbv")
    {
      #ifdef HAVE_FLOATBV
      float_utilst float_utils(prop);
      float_utils.spec=to_floatbv_type(expr.op0().type());
      return prop.land(
        prop.lnot(float_utils.is_infinity(bv)),
        prop.lnot(float_utils.is_NaN(bv)));
      #endif
    }
  }
  else if(expr.id()=="isinf")
  {
    if(expr.operands().size()!=1)
      throw "isinf expects one operand";

    bvt bv;
    convert_bv(operands[0], bv);
    
    if(expr.op0().type().id()=="floatbv")
    {
      #ifdef HAVE_FLOATBV
      float_utilst float_utils(prop);
      float_utils.spec=to_floatbv_type(expr.op0().type());
      return float_utils.is_infinity(bv);
      #endif
    }
  }
  else if(expr.id()=="isnormal")
  {
    if(expr.operands().size()!=1)
      throw "isnormal expects one operand";

    bvt bv;
    convert_bv(operands[0], bv);
    
    if(expr.op0().type().id()=="floatbv")
    {
      #ifdef HAVE_FLOATBV
      float_utilst float_utils(prop);
      float_utils.spec=to_floatbv_type(expr.op0().type());
      return float_utils.is_normal(bv);
      #endif
    }
  }

  return SUB::convert_rest(expr);
}

/*******************************************************************\

Function: boolbvt::boolbv_set_equality_to_true

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool boolbvt::boolbv_set_equality_to_true(const exprt &expr)
{
  if(!equality_propagation) return true;

  const exprt::operandst &operands=expr.operands();

  if(operands.size()==2)
  {
    if(operands[0].id()=="symbol" &&
       operands[0].type()==operands[1].type() &&
       operands[0].type().id()!="bool")
    {
      // see if it is an unbounded array
      if(is_unbounded_array(operands[0].type()))
        return true;

      bvt bv1;
      convert_bv(operands[1], bv1);
      
      const irep_idt &identifier=
        operands[0].get("identifier");

      const typet &type=operands[0].type();

      for(unsigned i=0; i<bv1.size(); i++)
        map.set_literal(identifier, i, type, bv1[i]);

      return false;
    }
  }

  return true;
}

/*******************************************************************\

Function: boolbvt::set_to

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::set_to(const exprt &expr, bool value)
{
  if(expr.type().id()!="bool")
  {
    std::cerr << expr << std::endl;
    throw "boolbvt::set_to got non-boolean operand";
  }

  if(value)
  {
    if(expr.id()=="=")
    {
      if(!boolbv_set_equality_to_true(expr))
        return;
    }
  }

  return SUB::set_to(expr, value);
}

/*******************************************************************\

Function: boolbvt::make_bv_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::make_bv_expr(const typet &type, const bvt &bv, exprt &dest)
{
  dest=exprt("bv_literals", type);
  irept::subt &bv_sub=dest.add("bv").get_sub();

  bv_sub.resize(bv.size());

  for(unsigned i=0; i<bv.size(); i++)
    bv_sub[i].id(i2string(bv[i].get()));
}

/*******************************************************************\

Function: boolbvt::make_bv_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::make_free_bv_expr(const typet &type, exprt &dest)
{
  unsigned width;
  if(boolbv_get_width(type, width))
    throw "failed to get width of "+type.to_string();

  bvt bv;
  bv.resize(width);

  // make result free variables
  for(unsigned i=0; i<bv.size(); i++)
    bv[i]=prop.new_variable();

  make_bv_expr(type, bv, dest);
}

/*******************************************************************\

Function: boolbvt::is_unbounded_array

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool boolbvt::is_unbounded_array(const typet &type) const
{
  if(type.id()!="array") return false;
  
  if(unbounded_array==U_ALL) return true;
  
  const exprt &size=(exprt &)type.find("size");
  
  mp_integer s;
  if(to_integer(size, s)) return true;

  if(unbounded_array==U_AUTO)
    if(s>1000) // magic number!
      return true;

  return false;
}

/*******************************************************************\

Function: boolbvt::print_assignment

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::print_assignment(std::ostream &out) const
{
  for(boolbv_mapt::mappingt::const_iterator it=map.mapping.begin();
      it!=map.mapping.end();
      it++)
  {
    out << it->first << "=";
    const boolbv_mapt::map_entryt &map_entry=it->second;

    std::string result="";
    for(unsigned i=0; i<map_entry.literal_map.size(); i++)
    {
      char ch='*';

      if(map_entry.literal_map[i].is_set)
      {
        tvt value=prop.l_get(map_entry.literal_map[i].l);
        if(value.is_true())
          ch='1';
        else if(value.is_false())
          ch='0';
        else
          ch='?';
      }
      
      result=result+ch;
    }
    
    out << result << std::endl;
  }
}

/*******************************************************************\

Function: boolbvt::build_offset_map

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::build_offset_map(const struct_typet &src, offset_mapt &dest)
{
  const struct_typet::componentst &components=
    src.components();

  dest.resize(components.size());
  unsigned offset=0;
  for(unsigned i=0; i<components.size(); i++)
  {
    unsigned int comp_width=0;
    boolbv_get_width(components[i].type(), comp_width);
    dest[i]=offset;
    offset+=comp_width;
  }
}
