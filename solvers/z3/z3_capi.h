/*******************************************************************\

Module:

Author:

\*******************************************************************/

#ifndef CPROVER_PROP_Z3_CAPI_H
#define CPROVER_PROP_Z3_CAPI_H

#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>
#include<memory.h>
#include<setjmp.h>
#include "z3++.h"

class z3_capi {

  public:

    z3_capi(){};  // constructor
    ~z3_capi(){}; // destructor

    void set_z3_ctx(Z3_context _ctx) { this->z3_ctx = _ctx; }

    Z3_lbool check2(Z3_lbool expected_result);

  private:
    Z3_context z3_ctx;
};

#endif
