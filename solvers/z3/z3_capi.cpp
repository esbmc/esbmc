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

void
throw_z3_error(Z3_context c __attribute__((unused)), Z3_error_code code)
{
  char buffer[16];

  snprintf(buffer, 15, "%d", code);
  buffer[15] = '\0';

  std::cout << "Z3 Error " << buffer << std::endl;
  abort();
}

Z3_context
z3_capi::mk_context_custom(Z3_config cfg, Z3_error_handler err)
{
  Z3_context ctx;

  Z3_set_param_value(cfg, "MODEL", "true");
  Z3_set_param_value(cfg, "RELEVANCY", "0");
  Z3_set_param_value(cfg, "SOLVER", "true");
  ctx = Z3_mk_context(cfg);
#ifdef TRACING
  Z3_trace_to_stderr(ctx);
#endif
  Z3_set_error_handler(ctx, err);

  return ctx;
}

Z3_context
z3_capi::mk_proof_context(void)
{
  Z3_config cfg = Z3_mk_config();
  Z3_context ctx;

  ctx = mk_context_custom(cfg, throw_z3_error);

  Z3_del_config(cfg);

  return ctx;
}

Z3_ast
z3_capi::mk_var(const char * name, Z3_sort ty) const
{
  Z3_symbol s  = Z3_mk_string_symbol(z3_ctx, name);
  return Z3_mk_const(z3_ctx, s, ty);
}

Z3_ast
z3_capi::mk_bool_var(const char * name)
{
  Z3_sort ty = Z3_mk_bool_sort(z3_ctx);
  return mk_var(name, ty);
}

Z3_ast
z3_capi::mk_int_var(const char * name)
{
  Z3_sort ty = Z3_mk_int_sort(z3_ctx);
  return mk_var(name, ty);
}

Z3_ast
z3_capi::mk_unsigned_int(unsigned int v)
{
  Z3_sort ty = Z3_mk_int_sort(z3_ctx);
  return Z3_mk_unsigned_int(z3_ctx, v, ty);
}

Z3_ast
z3_capi::mk_int(int v)
{
  Z3_sort ty = Z3_mk_int_sort(z3_ctx);
  return Z3_mk_int(z3_ctx, v, ty);
}

Z3_ast
z3_capi::mk_real_var(const char * name)
{
  Z3_sort ty = Z3_mk_real_sort(z3_ctx);
  return mk_var(name, ty);
}

Z3_ast
z3_capi::mk_unary_app(Z3_func_decl f, Z3_ast x)
{
  Z3_ast args[1] = {
    x
  };
  return Z3_mk_app(z3_ctx, f, 1, args);
}

Z3_ast
z3_capi::mk_binary_app(Z3_func_decl f, Z3_ast x, Z3_ast y)
{
  Z3_ast args[2] = {
    x, y
  };
  return Z3_mk_app(z3_ctx, f, 2, args);
}

Z3_ast
z3_capi::mk_tuple(Z3_sort sort, ...)
{
  va_list args;
  unsigned int num, i;

  // Count number of arguments
  va_start(args, sort);
  for (num = 0;; num++) {
    Z3_ast a = va_arg(args, Z3_ast);
    if (a == NULL)
      break;
  }
  va_end(args);

  // Generate array of args
  Z3_ast *arg_list = (Z3_ast*)alloca(sizeof(Z3_ast) * num);
  va_start(args, sort);
  for (i = 0;; i++) {
    Z3_ast a = va_arg(args, Z3_ast);
    if (a == NULL)
      break;
    arg_list[i] = a;
  }
  va_end(args);

  return mk_tuple(sort, arg_list, num);
}

Z3_ast
z3_capi::mk_tuple(Z3_sort sort, Z3_ast *args, unsigned int num)
{

  // Create appl
  Z3_func_decl decl = Z3_get_tuple_sort_mk_decl(z3_ctx, sort);
  Z3_ast val = Z3_mk_app(z3_ctx, decl, num, args);
  return val;
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
      new_fields[j] = mk_unary_app(proj_decl, t);
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
  return mk_unary_app(proj_decl, t);
}
