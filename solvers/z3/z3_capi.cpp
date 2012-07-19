#include <global.h>
#include <stdio.h>
#include <string>
#include <global.h>
#include <cstdarg>

#include "z3_conv.h"
#include "z3_capi.h"

void
abortf(const char* message)
{
  fprintf(stderr, "%s\n", message);
  abort();
}

Z3_ast
z3_capi::mk_tuple_update(Z3_ast t, unsigned i, Z3_ast new_val)
{
  Z3_sort ty;
  Z3_func_decl mk_tuple_decl;
  unsigned num_fields, j;
  Z3_ast *            new_fields;
  Z3_ast result;

  ty = Z3_get_sort(z3_ctx, t);

  if (Z3_get_sort_kind(z3_ctx, ty) != Z3_DATATYPE_SORT) {
    abortf("argument must be a tuple");
  }

  num_fields = Z3_get_tuple_sort_num_fields(z3_ctx, ty);

  if (i >= num_fields) {
    abortf("invalid tuple update, index is too big");
  }

  new_fields = (Z3_ast*) malloc(sizeof(Z3_ast) * num_fields);
  for (j = 0; j < num_fields; j++) {
    if (i == j) {
      /* use new_val at position i */
      new_fields[j] = new_val;
    } else   {
      /* use field j of t */
      Z3_func_decl proj_decl = Z3_get_tuple_sort_field_decl(z3_ctx, ty, j);
      Z3_ast args[1] = { t };
      new_fields[j] = Z3_mk_app(z3_ctx, proj_decl, 1, args);
    }
  }
  mk_tuple_decl = Z3_get_tuple_sort_mk_decl(z3_ctx, ty);
  result = Z3_mk_app(z3_ctx, mk_tuple_decl, num_fields, new_fields);
  free(new_fields);
  return result;
}

Z3_lbool
z3_capi::check2(Z3_lbool expected_result)
{
  Z3_model m      = 0;
  Z3_lbool result = Z3_check_and_get_model(z3_ctx, &m);
  switch (result) {
  case Z3_L_FALSE:
    break;

  case Z3_L_UNDEF:
    break;

  case Z3_L_TRUE:
    break;
  }
  if (m) {
    Z3_del_model(z3_ctx, m);
  }
  if (result != expected_result) {
    abortf("unexpected result");
  }

  return result;
}

Z3_ast
z3_capi::mk_tuple_select(Z3_ast t, unsigned i)
{
  Z3_sort ty;
  unsigned num_fields;

  ty = Z3_get_sort(z3_ctx, t);

  if (Z3_get_sort_kind(z3_ctx, ty) != Z3_DATATYPE_SORT) {
    throw new z3_convt::conv_error("argument must be a tuple");
  }

  num_fields = Z3_get_tuple_sort_num_fields(z3_ctx, ty);

  if (i >= num_fields) {
    throw new z3_convt::conv_error("invalid tuple select, index is too big");
  }

  Z3_func_decl proj_decl = Z3_get_tuple_sort_field_decl(z3_ctx, ty, i);
  Z3_ast args[1] = { t };
  return Z3_mk_app(z3_ctx, proj_decl, 1, args);
}
