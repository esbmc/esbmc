/*******************************************************************\

Module: SMT-LIB Builtin Logics, Finalizer for the QF_LIA logic 

Author: CM Wintersteiger

\*******************************************************************/

#ifndef SMT_FINALIZER_QF_LIA_H_
#define SMT_FINALIZER_QF_LIA_H_

#include "builtin_theories.h"

#include "smt_finalizer_generic.h"

class smt_typecheck_expr_QF_LIAt : public smt_typecheck_expr_generict
{
protected:  
  virtual typet search_fun(const exprt&, const typet&);
  virtual typet search_sort(const typet&);
  
public:
  smt_typecheck_expr_QF_LIAt(  message_streamt &eh, const exprt& b, 
                                const symbolst &t) : 
    smt_typecheck_expr_generict(eh, b, t)
    {};  
};

class smt_finalizer_QF_LIAt : public smt_finalizer_generict
{
  private:
    void check_double_sorts(const exprt&, const symbolst&, message_streamt&);
    void check_double_functions(const exprt&, const symbolst&, message_streamt&);    
    void check_double_predicates(
      const exprt&, const symbolst&, message_streamt&); 
       
  public:
    smt_finalizer_QF_LIAt(
      contextt &c,
      message_handlert &mh) :
        smt_finalizer_generict(c, mh)
      {};
    
    virtual ~smt_finalizer_QF_LIAt( void ) {};    
      
    virtual std::string logic( void ) const
      { return std::string("QF_LIA"); }
    
    virtual bool finalize( const exprt& );
};

#endif /*SMT_FINALIZER_QF_LIA_H_*/
