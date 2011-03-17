/*******************************************************************\

Module: SMT-LIB Builtin Logics, Finalizer for the AUFLIRA logic 

Author: CM Wintersteiger

\*******************************************************************/

#ifndef SMT_FINALIZER_AUFLIRA_H_
#define SMT_FINALIZER_AUFLIRA_H_

#include "builtin_theories.h"

#include "smt_finalizer_generic.h"

class smt_typecheck_expr_AUFLIRAt : public smt_typecheck_expr_generict
{
protected:  
  virtual typet search_fun(const exprt&, const typet&);
  virtual typet search_sort(const typet&);
  
public:
  smt_typecheck_expr_AUFLIRAt(  message_streamt &eh, const exprt& b, 
                                const symbolst &t) : 
    smt_typecheck_expr_generict(eh, b, t)
    {};  
};

class smt_finalizer_AUFLIRAt : public smt_finalizer_generict
{
  private:
    void check_double_sorts(const exprt&, const symbolst&, message_streamt&);
    void check_double_functions(const exprt&, const symbolst&, message_streamt&);    
    void check_double_predicates(
      const exprt&, const symbolst&, message_streamt&); 
       
  public:
    smt_finalizer_AUFLIRAt(
      contextt &c,
      message_handlert &mh) :
        smt_finalizer_generict(c, mh)
      {};
    
    virtual ~smt_finalizer_AUFLIRAt( void ) {};    
      
    virtual std::string logic( void ) const
      { return std::string("AUFLIRA"); }
    
    virtual bool finalize( const exprt& );
};

#endif /*SMT_FINALIZER_AUFLIRA_H_*/
