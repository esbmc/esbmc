/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com
Revision: Roberto Bruttomesso, roberto.bruttomesso@unisi.ch

\*******************************************************************/

#include <string.h>
#include <assert.h>
#include <ctype.h>

#include <fstream>

#include <arith_tools.h>
#include <std_types.h>
#include <config.h>
#include <i2string.h>
#include <expr_util.h>
#include <string2array.h>
#include <pointer_offset_size.h>
#include <find_symbols.h>

#include <solvers/flattening/boolbv_width.h>

#include "smt_conv.h"

/*******************************************************************\

Function: smt_convt::bin_zero

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string smt_convt::bin_zero(unsigned bits)
{
  assert(false && "Construct not supported yet");
  assert(bits!=0);
  std::string result="0bin";
  while(bits!=0) { result+='0'; bits--; }
  return result;
}

/*******************************************************************\

Function: smt_convt::smt_pointer_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string smt_convt::smt_pointer_type()
{
  assert(false && "Construct not supported yet");
  assert(config.ansi_c.pointer_width!=0);
  return "[# object: INT, offset: BITVECTOR("+
         i2string(config.ansi_c.pointer_width)+") #]";
}

/*******************************************************************\

Function: smt_convt::array_index_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string smt_convt::array_index_type()
{
  return std::string("32");
}

/*******************************************************************\

Function: smt_convt::array_index_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

typet smt_convt::gen_array_index_type()
{
  typet t("signedbv");
  t.width(32);
  return t;
}

/*******************************************************************\

Function: smt_convt::array_index

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string smt_convt::array_index(unsigned i)
{
  assert(false && "Construct not supported yet");
  return "0bin"+integer2binary(i, config.ansi_c.int_width);
}

/*******************************************************************\

Function: smt_convt::convert_address_of_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_convt::convert_address_of_rec(const exprt &expr)
{
  assert(false && "Construct not supported yet");

  if(expr.id()=="symbol" ||
     expr.is_constant() ||
     expr.id()=="string-constant")
  {
    smt_prop.out
      << "(# object:="
      << pointer_logic.add_object(expr)
      << ", offset:="
      << bin_zero(config.ansi_c.pointer_width) << " #)";
  }
  else if(expr.is_index())
  {
    if(expr.operands().size()!=2)
      throw "index takes two operands";

    const exprt &array=expr.op0();
    const exprt &index=expr.op1();

    if(index.is_zero())
    {
      if(array.type().id()=="pointer")
        convert_smt_expr(array);
      else if(array.type().is_array())
        convert_address_of_rec(array);
      else
        assert(false);
    }
    else
    {    
      smt_prop.out << "(LET P: ";
      smt_prop.out << smt_pointer_type();
      smt_prop.out << " = ";
      
      if(array.type().id()=="pointer")
        convert_smt_expr(array);
      else if(array.type().is_array())
        convert_address_of_rec(array);
      else
        assert(false);

      smt_prop.out << " IN P WITH .offset:=BVPLUS("
                   << config.ansi_c.pointer_width
                   << ", P.offset, ";
      convert_smt_expr(index);
      smt_prop.out << "))";
    }
  }
  else if(expr.id()=="member")
  {
    if(expr.operands().size()!=1)
      throw "member takes one operand";

    const exprt &struct_op=expr.op0();

    smt_prop.out << "(LET P: ";
    smt_prop.out << smt_pointer_type();
    smt_prop.out << " = ";
    
    convert_address_of_rec(struct_op);

    const irept::subt &components=
      struct_op.type().components().get_sub();
    
    const irep_idt &component_name=expr.component_name();
    
    bool found=false;
    
    mp_integer offset=1; // for the struct itself

    forall_irep(it, components)
    {
      if(component_name==it->name()) { found=true; break; }
      const typet &subtype=it->type();
      mp_integer sub_size=pointer_offset_size(subtype);
      if(sub_size==0) assert(false);
      offset+=sub_size;
    }
    
    assert(found);
    
    typet index_type("unsignedbv");
    index_type.width(config.ansi_c.pointer_width);

    exprt index=from_integer(offset, index_type);

    smt_prop.out << " IN P WITH .offset:=BVPLUS("
                 << config.ansi_c.pointer_width
                 << ", P.offset, ";
    convert_smt_expr(index);
    smt_prop.out << "))";
  }
  else
    throw "don't know how to take address of: "+expr.id_string();
}

/*******************************************************************\

Function: smt_convt::convert_rest

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt smt_convt::convert_rest(const exprt &expr)
{
  literalt l=prop.new_variable();
  
  find_symbols(expr);

  guards.push_back(std::make_pair(l, expr));

  return l;
}

/*******************************************************************\

Function: smt_convt::convert_identifier

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_convt::convert_identifier(const std::string &identifier)
{
  if (let_id.find(identifier)!=let_id.end())
    smt_prop.out << "?";
  if(flet_id.find(identifier )!=flet_id.end())
    smt_prop.out << "$";

  for(std::string::const_iterator
      it=identifier.begin();
      it!=identifier.end();
      it++)
  {
    char ch=*it;

    if(isalnum(ch) || ch=='$' || ch=='?')
      smt_prop.out << ch;
    else if(ch==':')
    {
      std::string::const_iterator next_it(it);
      next_it++;
      if(next_it!=identifier.end() && *next_it==':')
      {
        smt_prop.out << "__";
        it=next_it;
      }
      else
      {
        smt_prop.out << '_';
        smt_prop.out << int(ch);
        smt_prop.out << '_';
      }
    }
    else
    {
      smt_prop.out << '_';
      smt_prop.out << int(ch);
      smt_prop.out << '_';
    }
  }
}

/*******************************************************************\

Function: smt_convt::convert_as_bv

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_convt::convert_as_bv(const exprt &expr)
{
  if(expr.type().is_bool())
  {
    smt_prop.out << "IF ";
    convert_smt_expr(expr);
    smt_prop.out << " THEN 0bin1 ELSE 0bin0 ENDIF";
  }
  else
    convert_smt_expr(expr);
}

/*******************************************************************\

Function: smt_convt::convert_array_value

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_convt::convert_array_value(const exprt &expr)
{
  convert_as_bv(expr);
}

/*******************************************************************\

Function: smt_convt::convert_smt_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_convt::convert_smt_expr(const exprt &expr)
{
  if(expr.id()=="symbol")
  {
    convert_identifier(expr.identifier().as_string());
  }
  else if(expr.id()=="nondet_symbol")
  {
    convert_identifier("nondet"+expr.identifier().as_string());
  }
  else if(expr.id()=="typecast")
  {
    assert(false && "Construct not supported yet");
    assert(expr.operands().size()==1);
    const exprt &op=expr.op0();
    
    if(expr.type().is_bool())
    {
      if(op.type().id()=="signedbv" ||
         op.type().id()=="unsignedbv" ||
         op.type().id()=="pointer")
      {
        convert_smt_expr(op);
        smt_prop.out << "/=";
        convert_smt_expr(gen_zero(op.type()));
      }
      else
      {
        throw "TODO typecast1 "+op.type().id_string()+" -> bool";
      }
    }
    else if(expr.type().id()=="signedbv" ||
            expr.type().id()=="unsignedbv")
    {
      unsigned to_width=atoi(expr.type().width().c_str());
      
      if(op.type().id()=="signedbv")
      {
        unsigned from_width=atoi(op.type().width().c_str());
        
        if(from_width==to_width)
          convert_smt_expr(op);
        else if(from_width<to_width)
        {
          smt_prop.out << "SX(";
          convert_smt_expr(op);
          smt_prop.out << ", " << to_width << ")";
        }
        else
        {
          smt_prop.out << "(";
          convert_smt_expr(op);
          smt_prop.out << ")[" << (to_width-1) << ":0]";
        }
      }
      else if(op.type().id()=="unsignedbv")
      {
        unsigned from_width=atoi(op.type().width().c_str());
        
        if(from_width==to_width)
          convert_smt_expr(op);
        else if(from_width<to_width)
        {
          smt_prop.out << "(0bin";

          for(unsigned i=from_width; i<to_width; i++)
            smt_prop.out << "0";

          smt_prop.out << " @ ";
            
          smt_prop.out << "(";
          convert_smt_expr(op);
          smt_prop.out << "))";
        }
        else
        {
          smt_prop.out << "(";
          convert_smt_expr(op);
          smt_prop.out << ")[" << (to_width-1) << ":0]";
        }
      }
      else if(op.type().is_bool())
      {
        if(to_width>1)
        {
          smt_prop.out << "(0bin";

          for(unsigned i=1; i<to_width; i++)
            smt_prop.out << "0";

          smt_prop.out << " @ ";
          
          smt_prop.out << "IF ";
          convert_smt_expr(op);
          smt_prop.out << " THEN 0bin1 ELSE 0bin0 ENDIF)";
        }
        else
        {
          smt_prop.out << "IF ";
          convert_smt_expr(op);
          smt_prop.out << " THEN 0bin1 ELSE 0bin0 ENDIF";
        }
      }
      else
      {
        throw "TODO typecast2 "+op.type().id_string()+
              " -> "+expr.type().id_string();
      }
    }
    else if(expr.type().id()=="pointer")
    {
      if(op.type().id()=="pointer")
      {
        convert_smt_expr(op);
      }
      else
        throw "TODO typecast3 "+op.type().id_string()+" -> pointer";
    }
    else
      throw "TODO typecast4 ? -> "+expr.type().id_string();
  }
  else if(expr.id()=="struct")
  {
    assert(false && "Construct not supported yet");
    smt_prop.out << "(# ";
    
    const struct_typet &struct_type=to_struct_type(expr.type());
  
    const struct_typet::componentst &components=
      struct_type.components();
      
    assert(components.size()==expr.operands().size());

    unsigned i=0;
    for(struct_typet::componentst::const_iterator
        it=components.begin();
        it!=components.end();
        it++, i++)
    {
      if(i!=0) smt_prop.out << ", ";
      smt_prop.out << it->name();
      smt_prop.out << ":=";
      convert_smt_expr(expr.operands()[i]);
    }
    
    smt_prop.out << " #)";
  }
  else if(expr.is_constant())
  {
    if(expr.type().id()=="unsignedbv" ||
       expr.type().id()=="signedbv" ||
       expr.type().id()=="bv")
    {
      // RB: decimal conversion ... some solvers do not
      // support binary constants ...
      const char *str=expr.value().as_string().c_str();
      unsigned len=strlen(str);
      unsigned long value = 0;
      //
      // Conversion from binary to long ... hope it 
      // is enough, in any case we check
      //
      if (len>8*sizeof(value))
	throw "constant too big to fit a long";

      unsigned i = 0;
      // Move to the first one
      while ( i < len && str[ i++ ] == '0' );
      i--;

      for ( ; i < len ; i ++ )
      {
	value = value << 1;
	if ( str[ i ] == '1' )
	  value ++;
      }

      smt_prop.out << "bv" << value << "[" << len << "]";
    }
    else if(expr.type().id()=="pointer")
    {
      assert( false && "Construct not supported yet" );
      const irep_idt &value=expr.value();
      
      if(value=="NULL")
      {
        smt_prop.out << "(# object:="
                     << pointer_logic.get_null_object()
                     << ", offset:="
                     << bin_zero(config.ansi_c.pointer_width) << " #)";
      }
      else
        throw "unknown pointer constant: "+id2string(value);
    }
    else if(expr.type().is_bool())
    {
      if(expr.is_true())
        smt_prop.out << "true";
      else if(expr.is_false())
        smt_prop.out << "false";
      else
        throw "unknown boolean constant";
    }
    else if(expr.type().is_array())
    {
      assert( false && "Construct not supported yet" );
      smt_prop.out << "ARRAY (i: " << array_index_type() << "):";
      
      assert(expr.operands().size()!=0);
      
      unsigned i=0;
      forall_operands(it, expr)
      {
        if(i==0)
          smt_prop.out << "\n  IF ";
        else
          smt_prop.out << "\n  ELSIF ";

        smt_prop.out << "i=" << array_index(i) << " THEN ";
        convert_array_value(*it);
        i++;
      }
      
      smt_prop.out << "\n  ELSE ";
      convert_smt_expr(expr.op0());
      smt_prop.out << "\n  ENDIF";
    }
    else
    {
      assert( false && "Construct not supported yet" );
      std::cerr << expr.pretty() << std::endl;
      throw "unknown constant: "+expr.type().id_string();
    }
  }
  else if(expr.id()=="concatenation" || 
          expr.id()=="bitand" ||
          expr.id()=="bitor")
  {
    assert( false && "Construct not supported yet" );
    smt_prop.out << "(";

    forall_operands(it, expr)
    {
      if(it!=expr.operands().begin())
      {
        if(expr.id()=="concatenation")
          smt_prop.out << " @ ";
        else if(expr.id()=="bitand")
          smt_prop.out << " & ";
        else if(expr.id()=="bitor")
          smt_prop.out << " | ";
      }

      convert_as_bv(*it);
    }

    smt_prop.out << ")";
  }
  else if(expr.id()=="bitxor")
  {
    assert( false && "Construct not supported yet" );
    assert(!expr.operands().empty());
  
    if(expr.operands().size()==1)
    {
      convert_smt_expr(expr.op0());
    }
    else if(expr.operands().size()==2)
    {
      smt_prop.out << "BVXOR(";
      convert_smt_expr(expr.op0());
      smt_prop.out << ", ";
      convert_smt_expr(expr.op1());
      smt_prop.out << ")";
    }
    else
    {
      assert(expr.operands().size()>=3);
      
      exprt tmp(expr);
      tmp.operands().resize(tmp.operands().size()-1);

      smt_prop.out << "BVXOR(";
      convert_smt_expr(tmp);
      smt_prop.out << ", ";
      convert_smt_expr(expr.operands().back());
      smt_prop.out << ")";
    }
  }
  else if(expr.id()=="bitnand")
  {
    assert( false && "Construct not supported yet" );
    assert(expr.operands().size()==2);

    smt_prop.out << "BVNAND(";
    convert_smt_expr(expr.op0());
    smt_prop.out << ", ";
    convert_smt_expr(expr.op1());
    smt_prop.out << ")";
  }
  else if(expr.id()=="bitnot")
  {
    assert( false && "Construct not supported yet" );
    assert(expr.operands().size()==1);
    smt_prop.out << "~(";
    convert_smt_expr(expr.op0());
    smt_prop.out << ")";
  }
  else if(expr.id()=="unary-")
  {
    assert(false && "Construct not supported yet");
    assert(expr.operands().size()==1);
    if(expr.type().id()=="unsignedbv" ||
       expr.type().id()=="signedbv")
    {
      smt_prop.out << "BVUMINUS(";
      convert_smt_expr(expr.op0());
      smt_prop.out << ")";
    }
    else
      throw "unsupported type for unary-: "+expr.type().id_string();
  }
  else if(expr.id()=="if")
  {
    assert(expr.operands().size()==3);
    // smt_prop.out << "(if_then_else ";
    smt_prop.out << "(ite ";
    convert_smt_expr(expr.op0());
    smt_prop.out << " ";
    convert_smt_expr(expr.op1());
    smt_prop.out << " ";
    convert_smt_expr(expr.op2());
    smt_prop.out << ")";
  }
  else if(expr.is_and() ||
          expr.id()=="or" ||
          expr.id()=="xor")
  {
    assert(false && "Construct not supported yet");
    assert(expr.type().is_bool());
    
    if(expr.operands().size()>=2)
    {
      forall_operands(it, expr)
      {
        if(it!=expr.operands().begin())
        {
          if(expr.is_and())
            smt_prop.out << " AND ";
          else if(expr.id()=="or")
            smt_prop.out << " OR ";
          else if(expr.id()=="xor")
            smt_prop.out << " XOR ";
        }
        
        smt_prop.out << "(";
        convert_smt_expr(*it);
        smt_prop.out << ")";
      }
    }
    else if(expr.operands().size()==1)
    {
      convert_smt_expr(expr.op0());
    }
    else
      assert(false);
  }
  else if(expr.id()=="not")
  {
    assert(expr.operands().size()==1);
    smt_prop.out << "(not ";
    convert_smt_expr(expr.op0());
    smt_prop.out << ")";
  }
  else if(expr.id()=="=" ||
          expr.id()=="notequal")
  {
    assert(expr.operands().size()==2);
    assert(expr.op0().type()==expr.op1().type());

    if(expr.op0().type().is_bool())
    {
      if(expr.id()=="notequal") 
	smt_prop.out << "(xor ";
      else 
	smt_prop.out << "(iff ";

      convert_smt_expr(expr.op0());
      smt_prop.out << " ";
      convert_smt_expr(expr.op1());
      smt_prop.out << ")";
    }
    else
    {
      if(expr.id()=="notequal") 
	smt_prop.out << "(not ";
      smt_prop.out << "(= ";
      convert_smt_expr(expr.op0());
      smt_prop.out << " ";
      convert_smt_expr(expr.op1());
      smt_prop.out << ")";
      if(expr.id()=="notequal") 
	smt_prop.out << ")";
    }
  }
  else if(expr.id()=="<=" ||
          expr.id()=="<" ||
          expr.id()==">=" ||
          expr.id()==">")
  {
    assert(expr.operands().size()==2);
    
    const typet &op_type=expr.op0().type();

    if(op_type.id()=="unsignedbv")
    {
      smt_prop.out << "(";

      if(expr.id()=="<=")
        smt_prop.out << "bvuleq";
      else if(expr.id()=="<")
        smt_prop.out << "bvult";
      else if(expr.id()==">=")
        smt_prop.out << "bvugeq";
      else if(expr.id()==">")
        smt_prop.out << "bvugt";
      
      smt_prop.out << " ";
      convert_smt_expr(expr.op0());
      smt_prop.out << " ";
      convert_smt_expr(expr.op1());
      smt_prop.out << ")";
    }
    else if(op_type.id()=="signedbv")
    {
      smt_prop.out << "(";

      if(expr.id()=="<=")
        smt_prop.out << "bvsleq";
      else if(expr.id()=="<")
        smt_prop.out << "bvslt";
      else if(expr.id()==">=")
        smt_prop.out << "bvsgeq";
      else if(expr.id()==">")
        smt_prop.out << "bvsgt";
      
      smt_prop.out << " ";
      convert_smt_expr(expr.op0());
      smt_prop.out << " ";
      convert_smt_expr(expr.op1());
      smt_prop.out << ")";
    }
    else
      throw "unsupported type for "+expr.id_string()+": "+expr.type().id_string();
  }
  else if(expr.id()=="+")
  {
    if(expr.operands().size()>=2)
    {
      if(expr.type().id()=="unsignedbv" ||
	 expr.type().id()=="signedbv")
      {
        smt_prop.out << "(bvadd ";

        forall_operands(it, expr)
        {
          smt_prop.out << " ";
          convert_smt_expr(*it);
        }
          
        smt_prop.out << ")";
      }
      else if(expr.type().id()=="pointer")
      {
        if(expr.operands().size()!=2)
          throw "pointer arithmetic with more than two operands";
        
        const exprt *p, *i;
        
        if(expr.op0().type().id()=="pointer")
        {
          p=&expr.op0();
          i=&expr.op1();
        }
        else if(expr.op1().type().id()=="pointer")
        {
          p=&expr.op1();
          i=&expr.op0();
        }
        else
          throw "unexpected mixture in pointer arithmetic";
        
        smt_prop.out << "(LET P: " << smt_pointer_type() << " = ";
        convert_smt_expr(*p);
        smt_prop.out << " IN P WITH .offset:=BVPLUS("
                     << config.ansi_c.pointer_width
                     << ", P.offset, ";
        convert_smt_expr(*i);
        smt_prop.out << "))";
      }
      else
        throw "unsupported type for +: "+expr.type().id_string();
    }
    else if(expr.operands().size()==1)
    {
      convert_smt_expr(expr.op0());
    }
    else
      assert(false);
  }
  else if(expr.id()=="-")
  {
    assert(false && "Construct not supported yet");
    if(expr.operands().size()==2)
    {
      if(expr.type().id()=="unsignedbv" ||
         expr.type().id()=="signedbv")
      {
        smt_prop.out << "BVSUB(" << expr.type().width() << ", ";
        convert_smt_expr(expr.op0());
        smt_prop.out << ", ";
        convert_smt_expr(expr.op1());
        smt_prop.out << ")";
      }
      else
        throw "unsupported type for -: "+expr.type().id_string();
    }
    else if(expr.operands().size()==1)
    {
      convert_smt_expr(expr.op0());
    }
    else
      assert(false);
  }
  else if(expr.id()=="/")
  {
    assert(false && "Construct not supported yet");
    assert(expr.operands().size()==2);

    if(expr.type().id()=="unsignedbv" ||
       expr.type().id()=="signedbv")
    {
      if(expr.type().id()=="unsignedbv")
        smt_prop.out << "BVDIV";
      else
        smt_prop.out << "SBVDIV";

      smt_prop.out << "(" << expr.type().width() << ", ";
      convert_smt_expr(expr.op0());
      smt_prop.out << ", ";
      convert_smt_expr(expr.op1());
      smt_prop.out << ")";
    }
    else
      throw "unsupported type for /: "+expr.type().id_string();
  }
  else if(expr.id()=="mod")
  {
    assert(false && "Construct not supported yet");
    assert(expr.operands().size()==2);

    if(expr.type().id()=="unsignedbv" ||
       expr.type().id()=="signedbv")
    {
      if(expr.type().id()=="unsignedbv")
        smt_prop.out << "BVMOD";
      else
        smt_prop.out << "SBVMOD";

      smt_prop.out << "(" << expr.type().width() << ", ";
      convert_smt_expr(expr.op0());
      smt_prop.out << ", ";
      convert_smt_expr(expr.op1());
      smt_prop.out << ")";
    }
    else
      throw "unsupported type for mod: "+expr.type().id_string();
  }
  else if(expr.id()=="*")
  {
    assert(false && "Construct not supported yet");
    if(expr.operands().size()==2)
    {
      if(expr.type().id()=="unsignedbv" ||
         expr.type().id()=="signedbv")
      {
        smt_prop.out << "BVMULT(" << expr.type().width() << ", ";
        convert_smt_expr(expr.op0());
        smt_prop.out << ", ";
        convert_smt_expr(expr.op1());
        smt_prop.out << ")";
      }
      else
        throw "unsupported type for *: "+expr.type().id_string();
    }
    else if(expr.operands().size()==1)
    {
      convert_smt_expr(expr.op0());
    }
    else
      assert(false);
  }
  else if(expr.is_address_of() ||
          expr.id()=="implicit_address_of" ||
          expr.id()=="reference_to")
  {
    assert(false && "Construct not supported yet");
    assert(expr.operands().size()==1);
    assert(expr.type().id()=="pointer");
    convert_address_of_rec(expr.op0());
  }
  else if(expr.id()=="array_of")
  {
    //
    // Not supported but it does not work otherwise
    //
    // assert(false && "Construct not supported yet");
    assert(expr.type().is_array());
    assert(expr.operands().size()==1);
    smt_prop.out << "bv0[32]";
    // smt_prop.out << "(ARRAY (i: " << array_index_type() << "): ";
    // convert_array_value(expr.op0());
    // smt_prop.out << ")";
  }
  else if(expr.is_index())
  {
    assert(expr.operands().size()==2);
    smt_prop.out << "(select ";
    convert_smt_expr(expr.op0());
    smt_prop.out << " ";

    if(expr.op1().type()==gen_array_index_type())
    {
      convert_smt_expr(expr.op1());
    }
    else
    {
      exprt tmp("typecast", gen_array_index_type());
      tmp.copy_to_operands(expr.op1());
      convert_smt_expr(tmp);
    }
    
    smt_prop.out << ")";
  }
  else if(expr.id()=="ashr" ||
          expr.id()=="lshr" ||
          expr.id()=="shl")
  {
    assert(false && "Construct not supported yet");
    assert(expr.operands().size()==2);

    if(expr.type().id()=="unsignedbv" ||
       expr.type().id()=="signedbv")
    {
      if(expr.id()=="ashr")
        smt_prop.out << "BVASHR";
      else if(expr.id()=="lshr")
        smt_prop.out << "BVLSHR";
      else if(expr.id()=="shl")
        smt_prop.out << "BVSHL";
      else
        assert(false);

      smt_prop.out << "(" << expr.type().width() << ", ";
      convert_smt_expr(expr.op0());
      smt_prop.out << ", ";
      convert_smt_expr(expr.op1());
      smt_prop.out << ")";
    }
    else
      throw "unsupported type for "+expr.id_string()+": "+expr.type().id_string();
  }
  else if(expr.id()=="with")
  {
    assert(expr.operands().size()>=1);
    smt_prop.out << "(store ";
    convert_smt_expr(expr.op0());
  
    for(unsigned i=1; i<expr.operands().size(); i+=2)
    {
      assert((i+1)<expr.operands().size());
      const exprt &index=expr.operands()[i];
      const exprt &value=expr.operands()[i+1];

      if(expr.type().id()=="struct")
      {
	assert(false && "operator not supported yet");
        smt_prop.out << " WITH ." << index.component_name();
        smt_prop.out << ":=(";
        convert_array_value(value);
        smt_prop.out << ")";
      }
      else if(expr.type().id()=="union")
      {
	assert(false && "operator not supported yet");
        smt_prop.out << " WITH ." << index.component_name();
        smt_prop.out << ":=(";
        convert_array_value(value);
        smt_prop.out << ")";
      }
      else if(expr.type().is_array())
      {
        smt_prop.out << " ";
        convert_smt_expr(index);
        smt_prop.out << " ";
        convert_array_value(value);
      }
      else
        throw "with expects struct or array type, but got "+expr.type().id_string();
    }

    smt_prop.out << ")";
  }
  else if(expr.id()=="member")
  {
    assert(false && "Construct not supported yet");
    assert(expr.operands().size()==1);
    convert_smt_expr(expr.op0());
    smt_prop.out << ".";
    smt_prop.out << expr.component_name();
  }
  else if(expr.id()=="pointer_offset")
  {
    assert(false && "Construct not supported yet");
    assert(expr.operands().size()==1);
    smt_prop.out << "(";
    convert_smt_expr(expr.op0());
    smt_prop.out << ").offset";
  }
  #if 0
  else if(expr.id()=="pointer_object")
  {
    assert(expr.operands().size()==1);
    smt_prop.out << "(";
    convert_smt_expr(expr.op0());
    smt_prop.out << ").object";
    // TODO, this has the wrong type
  }
  #endif
  else if(expr.id()=="same-object")
  {
    assert(false && "Construct not supported yet");
    assert(expr.operands().size()==2);
    smt_prop.out << "(";
    convert_smt_expr(expr.op0());
    smt_prop.out << ").object=(";
    convert_smt_expr(expr.op1());
    smt_prop.out << ").object";
  }
  else if(expr.id()=="string-constant")
  {
    assert(false && "Construct not supported yet");
    exprt tmp;
    string2array(expr, tmp);
    convert_smt_expr(tmp);
  }
  else if(expr.id()=="extractbit")
  {
    assert(false && "Construct not supported yet");
    assert(expr.operands().size()==2);

    if(expr.op0().type().id()=="unsignedbv" ||
       expr.op0().type().id()=="signedbv")
    {
      smt_prop.out << "(";
      convert_smt_expr(expr.op0());
      
      mp_integer i;
      if(to_integer(expr.op1(), i))
        throw "extractbit takes constant as second parameter";
        
      smt_prop.out << "[" << i << ":" << i << "]=0bin1)";
    }
    else
      throw "unsupported type for "+expr.id_string()+": "+expr.op0().type().id_string();
  }
  else if(expr.id()=="replication")
  {
    assert(false && "Construct not supported yet");
    assert(expr.operands().size()==2);
  
    mp_integer times;
    if(to_integer(expr.op0(), times))
      throw "replication takes constant as first parameter";
    
    smt_prop.out << "(LET v: BITVECTOR(1) = ";

    convert_smt_expr(expr.op1());

    smt_prop.out << " IN ";

    for(mp_integer i=0; i<times; ++i)
    {
      if(i!=0) smt_prop.out << "@";
      smt_prop.out << "v";
    }
    
    smt_prop.out << ")";
  }
  else
    throw "convert_smt_expr: "+expr.id_string()+" is unsupported";
}

/*******************************************************************\

Function: smt_convt::set_to

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_convt::set_to(const exprt &expr, bool value)
{
  if(value && expr.is_and())
  {
    forall_operands(it, expr)
      set_to(*it, true);
    return;
  }
  
  if(value && expr.is_true())
    return;

  // smt_prop.out << "%% set_to " << (value?"true":"false") << std::endl;

  if(expr.id()=="=" && value)
  {
    assert(expr.operands().size()==2);
    
    if(expr.op0().id()=="symbol")
    {
      const irep_idt &identifier=expr.op0().identifier();
      
      identifiert &id=identifier_map[identifier];

      if(id.type.is_nil())
      {
        hash_set_cont<irep_idt, irep_id_hash> s_set;

        ::find_symbols(expr.op1(), s_set);

        if(s_set.find(identifier)==s_set.end())
        {
          id.type=expr.op0().type();

          find_symbols(expr.op1());

	  // Store definition
	  defines.push_back(std::make_pair( expr, expr.op0().type().id()!="bool" ));

          return;
        }
      }
    }
  }

  find_symbols(expr);

  assumptions.push_back(std::make_pair(expr, value));
}

/*******************************************************************\

Function: smt_convt::find_symbols

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_convt::find_symbols(const exprt &expr)
{
  find_symbols(expr.type());

  forall_operands(it, expr)
    find_symbols(*it);
    
  if(expr.id()=="symbol")
  {
    if(expr.type().is_code())
      return;

    const irep_idt &identifier=expr.identifier();
    identifiert &id=identifier_map[identifier];

    if(id.type.is_nil())
    {
      id.type=expr.type();

      smt_prop.out << ":extrafuns(( ";
      convert_identifier(id2string(identifier));
      smt_prop.out << " ";
      convert_smt_type(expr.type());
      smt_prop.out << ")) " << std::endl;
    }
  }  
  else if(expr.id()=="nondet_symbol")
  {
    if(expr.type().is_code())
      return;

    const irep_idt identifier="nondet"+expr.identifier().as_string();

    identifiert &id=identifier_map[identifier];

    if(id.type.is_nil())
    {
      id.type=expr.type();

      if(expr.type().id()!="bool")
	smt_prop.out << ":extrafuns(( ";
      else
	smt_prop.out << ":extrapreds(( ";

      convert_identifier(id2string(identifier));
      smt_prop.out << " ";
      
      if(expr.type().id()!="bool")
	convert_smt_type(expr.type());

      smt_prop.out << ")) " << std::endl;
    }
  }  
}

/*******************************************************************\

Function: smt_convt::convert_smt_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_convt::convert_smt_type(const typet &type)
{
  if(type.is_array())
  {
    const array_typet &array_type=to_array_type(type);
    
    smt_prop.out << "Array[32:";
                 
    if(array_type.subtype().is_bool())
      smt_prop.out << "1";
    else
      smt_prop.out << "32";

    smt_prop.out << "]";
  }
  /*
  else if(type.is_bool())
  {
    assert(false && "Construct not supported yet");
    smt_prop.out << "BOOLEAN";
  }
  */
  else if(type.id()=="struct" ||
          type.id()=="union")
  {
    assert(false && "Construct not supported yet");
    const struct_typet &struct_type=to_struct_type(type);
  
    smt_prop.out << "[#";
    
    const struct_typet::componentst &components=
      struct_type.components();

    for(struct_typet::componentst::const_iterator
        it=components.begin();
        it!=components.end();
        it++)
    {
      if(it!=components.begin()) smt_prop.out << ",";
      smt_prop.out << " ";
      smt_prop.out << it->name();
      smt_prop.out << ": ";
      convert_smt_type(it->type());
    }
    
    smt_prop.out << " #]";
  }
  else if(type.id()=="pointer" ||
          type.id()=="reference")
  {
    assert(false && "Construct not supported yet");
    smt_prop.out << smt_pointer_type();
  }
  else if(type.id()=="integer")
  {
    smt_prop.out << "Int";
  }
  else
  {
    unsigned width;

    if(boolbv_get_width(type, width))
      throw "unsupported type: "+type.id_string();
      
    if(width==0)
      throw "zero-width vector type: "+type.id_string();
  
    smt_prop.out << "BitVec[" << width << "]";
  }
}    

/*******************************************************************\

Function: smt_convt::find_symbols

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smt_convt::find_symbols(const typet &type)
{
  if(type.is_array())
  {
    const array_typet &array_type=to_array_type(type);
    find_symbols(array_type.size());
  }
  else if(type.id()=="struct" ||
          type.id()=="union")
  {
  }
}
