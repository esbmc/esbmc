/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com
Revision: Roberto Bruttomesso, roberto.bruttomesso@unisi.ch

\*******************************************************************/

#ifndef CPROVER_SOLVER_SMT_CONV_H
#define CPROVER_SOLVER_SMT_CONV_H

#include <sstream>
#include <set>

#include <hash_cont.h>

#include <solvers/prop/prop_conv.h>
#include <solvers/flattening/pointer_logic.h>

#include "smt_prop.h"

class smt_prop_wrappert
{
public:
  smt_prop_wrappert(std::ostream &_out):smt_prop(_out) 
  { }

protected:
  smt_propt smt_prop;
};

class smt_convt:protected smt_prop_wrappert, public prop_convt
{
public:
  smt_convt(std::ostream &_out)
    : smt_prop_wrappert(_out)
    , prop_convt(smt_prop) 
  { }

  virtual ~smt_convt() { }

protected:
  virtual literalt convert_rest(const exprt &expr);
  virtual void convert_smt_expr(const exprt &expr);
  virtual void convert_smt_type(const typet &type);
  virtual void set_to(const exprt &expr, bool value);
  virtual void convert_address_of_rec(const exprt &expr);

  pointer_logict pointer_logic;

  std::vector< std::pair< exprt, bool > >     assumptions;
  std::vector< std::pair< literalt, exprt > > guards;
  std::vector< std::pair< exprt, bool > >     defines;
  std::set< std::string >		      let_id;
  std::set< std::string >		      flet_id;

private:
  void convert_identifier(const std::string &identifier);
  void find_symbols(const exprt &expr);
  void find_symbols(const typet &type);
  void convert_array_value(const exprt &expr);
  void convert_as_bv(const exprt &expr);
  static typet gen_array_index_type();
  static std::string bin_zero(unsigned bits);
  static std::string array_index_type();
  static std::string array_index(unsigned i);
  static std::string smt_pointer_type();
  
  struct identifiert
  {
    typet type;
    exprt value;
    
    identifiert()
    {
      type.make_nil();
      value.make_nil();
    }
  };
  
  typedef hash_map_cont<irep_idt, identifiert, irep_id_hash>
    identifier_mapt;

  identifier_mapt identifier_map;
};

#endif
