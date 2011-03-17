/*******************************************************************\

Module: SMT-LIB Builtin Logics, Finalizer Base Class 

Author: CM Wintersteiger

\*******************************************************************/

#ifndef SMT_FINALIZER_H_
#define SMT_FINALIZER_H_

#include <message_stream.h>
#include <context.h>
#include <message.h>

class smt_finalizert : public message_streamt
{
  protected:
    contextt &context;    
    
  public:
    smt_finalizert(
      contextt &c,
      message_handlert &mh) :
        message_streamt(mh),
        context(c)        
      {};
      
    virtual ~smt_finalizert( void ) {};
    
    virtual std::string logic( void ) const = 0;
    
    virtual bool finalize( const exprt& ) = 0;
};

#endif /*SMT_FINALIZER_H_*/
