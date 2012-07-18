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

    Z3_ast mk_var(const char * name, Z3_sort ty) const;
    Z3_ast mk_unary_app(Z3_func_decl f, Z3_ast x);
    Z3_ast mk_binary_app(Z3_func_decl f, Z3_ast x, Z3_ast y);
    Z3_ast mk_tuple(Z3_sort tuple_sort, ...);
    Z3_ast mk_tuple(Z3_sort tuple_sort, Z3_ast *args,
                    unsigned int num);
    Z3_lbool check2(Z3_lbool expected_result);
    Z3_ast mk_tuple_update(Z3_ast t, unsigned i, Z3_ast new_val);
    Z3_ast mk_tuple_select(Z3_ast t, unsigned i);

  private:
    Z3_context z3_ctx;
};

#endif
