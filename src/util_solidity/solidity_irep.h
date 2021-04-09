/*******************************************************************\

Module:

Author: Kunjian Song

\*******************************************************************/

#ifndef SOLIDITY_AST_FRONTEND_SOLIDITY_IREP_H
#define SOLIDITY_AST_FRONTEND_SOLIDITY_IREP_H

#include <cassert>
//#include <list>
//#include <map>
//#include <string>
//#include <vector>

//#define USE_DSTRING
//#define SHARING

//#include <util/dstring.h>

#if 0
#define forall_irep(it, irep)                                                  \
  for(irept::subt::const_iterator it = (irep).begin(); it != (irep).end(); it++)

#define Forall_irep(it, irep)                                                  \
  for(irept::subt::iterator it = (irep).begin(); it != (irep).end(); it++)

#define forall_named_irep(it, irep)                                            \
  for(irept::named_subt::const_iterator it = (irep).begin();                   \
      it != (irep).end();                                                      \
      it++)

#define Forall_named_irep(it, irep)                                            \
  for(irept::named_subt::iterator it = (irep).begin(); it != (irep).end(); it++)
#endif

#include <iostream>

class solidity_irept
{
public:
    solidity_irept() { };
};

//const irept &get_nil_irep();

#endif
