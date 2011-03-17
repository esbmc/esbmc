/*******************************************************************\

Module: SMT-LIB Builtin Logics, Generic Finalizer 

Author: CM Wintersteiger

\*******************************************************************/

#ifndef SMT_FINALIZER_GENERIC_H_
#define SMT_FINALIZER_GENERIC_H_

#include <string>

#include "smt_typecheck_expr.h"
#include "smt_finalizer.h"

class smt_typecheck_expr_generict : public smt_typecheck_exprt
{
protected:  
  virtual typet search_fun(const exprt&, const typet&);
  virtual typet search_sort(const typet&);
    
public:
  smt_typecheck_expr_generict(  message_streamt &eh, const exprt& b, 
                                const symbolst &t) : 
    smt_typecheck_exprt(eh, b, t)
    {};  
};

class smt_finalizer_generict : public smt_finalizert
{
  private:
    void check_double_sorts(const exprt&, const symbolst&, message_streamt&);
    void check_double_functions(const exprt&, const symbolst&, message_streamt&);
    void check_double_function_signatures(
      const exprt&, const typet&, const typet&, message_streamt&);
    void check_double_predicates(
      const exprt&, const symbolst&, message_streamt&); 
       
  public:
    smt_finalizer_generict(
      contextt &c,
      message_handlert &mh) :
        smt_finalizert(c, mh)
      {};
    
    virtual ~smt_finalizer_generict( void ) {};
      
    virtual std::string logic( void ) const
      { return std::string("Generic"); }
    
    virtual bool finalize( const exprt& );
};

#endif /*SMT_FINALIZER_GENERIC_H_*/
