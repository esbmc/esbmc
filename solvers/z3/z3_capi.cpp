/*******************************************************************\

Module:

Author:

\*******************************************************************/

#include <global.h>
#include <stdio.h>
#include <string>
#include <global.h>
#include <cstdarg>

#include "z3_conv.h"
#include "z3_capi.h"

/**
   \defgroup capi_ex C API examples
*/
/*@{*/
/**
   @name Auxiliary Functions
*/
/*@{*/

/**
   \brief exit gracefully in case of error.
*/
void exitf(const char* message)
{
  fprintf(stderr,"%s\n", message);
  exit(1);
}

/**
   \brief exit if unreachable code was reached.
*/
void unreachable()
{
    exitf("unreachable code was reached");
}

/**
   \brief Simpler error handler.
 */
void error_handler(Z3_error_code e)
{
	printf("Error code: %d\n", e);
    exitf("incorrect use of Z3");
}

/**
   \brief Low tech exceptions.

   In high-level programming languages, an error handler can throw an exception.
*/
void throw_z3_error(Z3_error_code c)
{
  char buffer[16];

  snprintf(buffer, 15, "%d", c);
  buffer[15] = '\0';

  std::cout << "Z3 Error " << buffer << std::endl;
  abort();
}

/**
   \brief Create a logical context.

   Enable model construction. Other configuration parameters can be passed in the cfg variable.

   Also enable tracing to stderr and register custom error handler.
*/
Z3_context z3_capi::mk_context_custom(Z3_config cfg, Z3_error_handler err)
{
    Z3_context ctx;

    Z3_set_param_value(cfg, "MODEL", "true");
    ctx = Z3_mk_context(cfg);
#ifdef TRACING
    Z3_trace_to_stderr(ctx);
#endif
    Z3_set_error_handler(ctx, err);

    return ctx;
}

Z3_context z3_capi::mk_proof_context(unsigned int is_uw)
{
  Z3_config cfg = Z3_mk_config();
  Z3_context ctx;

  if (is_uw)
  {
    Z3_set_param_value(cfg,"PROOF_MODE","0");
    Z3_set_param_value(cfg,"RELEVANCY","0");
  }
  else
  {
    Z3_set_param_value(cfg,"SOLVER","true");
	Z3_set_param_value(cfg,"RELEVANCY","0");
  }

  ctx = mk_context_custom(cfg,throw_z3_error);

  //Z3_open_log(ctx, "adpcm_encode_nopointer.log");
  //Z3_trace_to_file(ctx, "adpcm_encode_nopointer.trace");

  Z3_del_config(cfg);

  return ctx;
}

/**
   \brief Create a logical context.

   Enable model construction only.

   Also enable tracing to stderr and register standard error handler.
*/
Z3_context z3_capi::mk_context(char *solver)
{
    Z3_config  cfg;
    Z3_context ctx;
    cfg = Z3_mk_config();
    Z3_set_param_value(cfg,"RELEVANCY","0");
    Z3_set_param_value(cfg,"SOLVER",solver);
    ctx = mk_context_custom(cfg, error_handler);
    //Z3_open_log(ctx, "01_pthread10.log");
    //Z3_trace_to_file(ctx, "01_pthread10.trace");
    Z3_del_config(cfg);
    return ctx;
}

/**
   \brief Create a variable using the given name and type.
*/
Z3_ast z3_capi::mk_var(const char * name, Z3_sort ty)
{
    Z3_symbol   s  = Z3_mk_string_symbol(z3_ctx, name);
    return Z3_mk_const(z3_ctx, s, ty);
}

/**
   \brief Create a boolean variable using the given name.
*/
Z3_ast z3_capi::mk_bool_var(const char * name)
{
    Z3_sort ty = Z3_mk_bool_sort(z3_ctx);
    return mk_var(name, ty);
}

/**
   \brief Create an integer variable using the given name.
*/
Z3_ast z3_capi::mk_int_var(const char * name)
{
    Z3_sort ty = Z3_mk_int_sort(z3_ctx);
    return mk_var(name, ty);
}

/**
   \brief Create a Z3 integer node using a C int.
*/
Z3_ast z3_capi::mk_unsigned_int(unsigned int v)
{
    Z3_sort ty = Z3_mk_int_sort(z3_ctx);
    return Z3_mk_unsigned_int(z3_ctx, v, ty);
}

/**
   \brief Create a Z3 integer node using a C int.
*/
Z3_ast z3_capi::mk_int(int v)
{
    Z3_sort ty = Z3_mk_int_sort(z3_ctx);
    return Z3_mk_int(z3_ctx, v, ty);
}

/**
   \brief Create a real variable using the given name.
*/
Z3_ast z3_capi::mk_real_var(const char * name)
{
    Z3_sort ty = Z3_mk_real_sort(z3_ctx);
    return mk_var(name, ty);
}

/**
   \brief Create the unary function application: <tt>(f x)</tt>.
*/
Z3_ast z3_capi::mk_unary_app(Z3_func_decl f, Z3_ast x)
{
    Z3_ast args[1] = {x};
    return Z3_mk_app(z3_ctx, f, 1, args);
}

/**
   \brief Create the binary function application: <tt>(f x y)</tt>.
*/
Z3_ast z3_capi::mk_binary_app(Z3_func_decl f, Z3_ast x, Z3_ast y)
{
    Z3_ast args[2] = {x, y};
    return Z3_mk_app(z3_ctx, f, 2, args);
}

Z3_ast z3_capi::mk_tuple(Z3_sort sort, ...)
{
  va_list args;
  unsigned int num, i;

  // Count number of arguments
  va_start(args, sort);
  for (num = 0; ; num++) {
    Z3_ast a = va_arg(args, Z3_ast);
    if (a == NULL)
      break;
  }
  va_end(args);

  // Generate array of args
  Z3_ast *arg_list = (Z3_ast*)alloca(sizeof(Z3_ast) * num);
  va_start(args, sort);
  for (i = 0; ; i++) {
    Z3_ast a = va_arg(args, Z3_ast);
    if (a == NULL)
      break;
    arg_list[i] = a;
  }
  va_end(args);

  return mk_tuple(sort, arg_list, num);
}

Z3_ast z3_capi::mk_tuple(Z3_sort sort, Z3_ast *args,
                         unsigned int num)
{

  // Create appl
  Z3_func_decl decl = Z3_get_tuple_sort_mk_decl(z3_ctx, sort);
  Z3_ast val = Z3_mk_app(z3_ctx, decl, num, args);
  return val;
}

/**
   \brief Check whether the logical context is satisfiable, and compare the result with the expected result.
   If the context is satisfiable, then display the model.
*/
Z3_lbool z3_capi::check(Z3_lbool expected_result)
{
    Z3_model m      = 0;
    Z3_lbool result = Z3_check_and_get_model(z3_ctx, &m);
    switch (result) {
    case Z3_L_FALSE:
        printf("unsat\n");
        break;
    case Z3_L_UNDEF:
        printf("unknown\n");
        printf("potential model:\n%s\n", Z3_model_to_string(z3_ctx, m));
        break;
    case Z3_L_TRUE:
        printf("sat\n%s\n", Z3_model_to_string(z3_ctx, m));
        break;
    }
    if (m) {
        Z3_del_model(z3_ctx, m);
    }
    if (result != expected_result) {
        //exitf("unexpected result");
    }

    return result;
}

/**
   \brief Prove that the constraints already asserted into the logical
   context implies the given formula.  The result of the proof is
   displayed.

   Z3 is a satisfiability checker. So, one can prove \c f by showing
   that <tt>(not f)</tt> is unsatisfiable.

   The context \c ctx is not modified by this function.
*/
void z3_capi::prove(Z3_ast f, Z3_bool is_valid)
{
    Z3_model m;
    Z3_ast   not_f;

    /* save the current state of the context */
    Z3_push(z3_ctx);

    not_f = Z3_mk_not(z3_ctx, f);
    Z3_assert_cnstr(z3_ctx, not_f);

    m = 0;

    switch (Z3_check_and_get_model(z3_ctx, &m)) {
    case Z3_L_FALSE:
        /* proved */
        printf("valid\n");
        if (!is_valid) {
            //exitf("unexpected result");
        }
        break;
    case Z3_L_UNDEF:
        /* Z3 failed to prove/disprove f. */
        printf("unknown\n");
        if (m != 0) {
            /* m should be viewed as a potential counterexample. */
            printf("potential counterexample:\n%s\n", Z3_model_to_string(z3_ctx, m));
        }
        if (is_valid) {
            //exitf("unexpected result");
        }
        break;
    case Z3_L_TRUE:
        /* disproved */
        printf("invalid\n");
        if (m) {
            /* the model returned by Z3 is a counterexample */
            printf("counterexample:\n%s\n", Z3_model_to_string(z3_ctx, m));
        }
        if (is_valid) {
            //exitf("unexpected result");
        }
        break;
    }

    if (m) {
        Z3_del_model(z3_ctx, m);
    }

    /* restore context */
    Z3_pop(z3_ctx, 1);
}

/**
   \brief Assert the axiom: function f is injective in the i-th argument.

   The following axiom is asserted into the logical context:
   \code
   forall (x_0, ..., x_n) finv(f(x_0, ..., x_i, ..., x_{n-1})) = x_i
   \endcode

   Where, \c finv is a fresh function declaration.
*/
void z3_capi::assert_inj_axiom(Z3_func_decl f, unsigned i)
{
    unsigned sz, j;
    Z3_sort finv_domain, finv_range;
    Z3_func_decl finv;
    Z3_sort * types; /* types of the quantified variables */
    Z3_symbol *   names; /* names of the quantified variables */
    Z3_ast * xs;         /* arguments for the application f(x_0, ..., x_i, ..., x_{n-1}) */
    Z3_ast   x_i, fxs, finv_fxs, eq;
    Z3_pattern p;
    Z3_ast   q;
    sz = Z3_get_domain_size(z3_ctx, f);

    if (i >= sz) {
        exitf("failed to create inj axiom");
    }

    /* declare the i-th inverse of f: finv */
    finv_domain = Z3_get_range(z3_ctx, f);
    finv_range  = Z3_get_domain(z3_ctx, f, i);
    finv        = Z3_mk_fresh_func_decl(z3_ctx, "inv", 1, &finv_domain, finv_range);

    /* allocate temporary arrays */
    types       = (Z3_sort *) malloc(sizeof(Z3_sort) * sz);
    names       = (Z3_symbol *) malloc(sizeof(Z3_symbol) * sz);
    xs          = (Z3_ast *) malloc(sizeof(Z3_ast) * sz);

    /* fill types, names and xs */
    for (j = 0; j < sz; j++) { types[j] = Z3_get_domain(z3_ctx, f, j); };
    for (j = 0; j < sz; j++) { names[j] = Z3_mk_int_symbol(z3_ctx, j); };
    for (j = 0; j < sz; j++) { xs[j]    = Z3_mk_bound(z3_ctx, j, types[j]); };

    x_i = xs[i];

    /* create f(x_0, ..., x_i, ..., x_{n-1}) */
    fxs         = Z3_mk_app(z3_ctx, f, sz, xs);

    /* create f_inv(f(x_0, ..., x_i, ..., x_{n-1})) */
    finv_fxs    = mk_unary_app(finv, fxs);

    /* create finv(f(x_0, ..., x_i, ..., x_{n-1})) = x_i */
    eq          = Z3_mk_eq(z3_ctx, finv_fxs, x_i);

    /* use f(x_0, ..., x_i, ..., x_{n-1}) as the pattern for the quantifier */
    p           = Z3_mk_pattern(z3_ctx, 1, &fxs);
    printf("pattern: %s\n", Z3_pattern_to_string(z3_ctx, p));
    printf("\n");

    /* create & assert quantifier */
    q           = Z3_mk_forall(z3_ctx,
                                 0, /* using default weight */
                                 1, /* number of patterns */
                                 &p, /* address of the "array" of patterns */
                                 sz, /* number of quantified variables */
                                 types,
                                 names,
                                 eq);
    printf("assert axiom:\n%s\n", Z3_ast_to_string(z3_ctx, q));
    Z3_assert_cnstr(z3_ctx, q);

    /* free temporary arrays */
    free(types);
    free(names);
    free(xs);
}

/**
   \brief Assert the axiom: function f is commutative.

   This example uses the SMT-LIB parser to simplify the axiom construction.
*/
void z3_capi::assert_comm_axiom(Z3_func_decl f)
{
    Z3_sort t;
    Z3_symbol f_name, t_name;
    Z3_ast q;

    t = Z3_get_range(z3_ctx, f);

    if (Z3_get_domain_size(z3_ctx, f) != 2 ||
        Z3_get_domain(z3_ctx, f, 0) != t ||
        Z3_get_domain(z3_ctx, f, 1) != t) {
        exitf("function must be binary, and argument types must be equal to return type");
    }

    /* Inside the parser, function f will be referenced using the symbol 'f'. */
    f_name = Z3_mk_string_symbol(z3_ctx, "f");

    /* Inside the parser, type t will be referenced using the symbol 'T'. */
    t_name = Z3_mk_string_symbol(z3_ctx, "T");

    Z3_parse_smtlib_string(z3_ctx,
                           "(benchmark comm :formula (forall (x T) (y T) (= (f x y) (f y x))))",
                           1, &t_name, &t,
                           1, &f_name, &f);
    q = Z3_get_smtlib_formula(z3_ctx, 0);
    printf("assert axiom:\n%s\n", Z3_ast_to_string(z3_ctx, q));
    Z3_assert_cnstr(z3_ctx, q);
}

/**
   \brief Z3 does not support explicitly tuple updates. They can be easily implemented
   as macros. The argument \c t must have tuple type.
   A tuple update is a new tuple where field \c i has value \c new_val, and all
   other fields have the value of the respective field of \c t.

   <tt>update(t, i, new_val)</tt> is equivalent to
   <tt>mk_tuple(proj_0(t), ..., new_val, ..., proj_n(t))</tt>
*/
Z3_ast z3_capi::mk_tuple_update(Z3_ast t, unsigned i, Z3_ast new_val)
{
    Z3_sort         ty;
    Z3_func_decl   mk_tuple_decl;
    unsigned            num_fields, j;
    Z3_ast *            new_fields;
    Z3_ast              result;

    ty = Z3_get_sort(z3_ctx, t);

    if (Z3_get_sort_kind(z3_ctx, ty) != Z3_DATATYPE_SORT) {
        exitf("argument must be a tuple");
    }

    num_fields = Z3_get_tuple_sort_num_fields(z3_ctx, ty);

    if (i >= num_fields) {
        exitf("invalid tuple update, index is too big");
    }

    new_fields = (Z3_ast*) malloc(sizeof(Z3_ast) * num_fields);
    for (j = 0; j < num_fields; j++) {
        if (i == j) {
            /* use new_val at position i */
            new_fields[j] = new_val;
        }
        else {
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

/**
   \brief Similar to #check, but uses #display_model instead of #Z3_model_to_string.
*/
Z3_lbool z3_capi::check2(Z3_lbool expected_result)
{
	Z3_model m      = 0;
    Z3_lbool result = Z3_check_and_get_model(z3_ctx, &m);
    switch (result) {
    case Z3_L_FALSE:
        //printf("unsat\n");
        break;
    case Z3_L_UNDEF:
        //printf("unknown\n");
        //printf("potential model:\n");
        //display_model(stdout, m);
        break;
    case Z3_L_TRUE:
        //printf("sat\n");
    	printf("Counterexample:\n");
        //display_model(stdout, m);
        break;
    }
    if (m) {
        Z3_del_model(z3_ctx, m);
    }
    if (result != expected_result) {
        //exitf("unexpected result");
    }

    return result;
}

Z3_ast z3_capi::mk_tuple_select(Z3_ast t, unsigned i)
{
    Z3_type_ast         ty;
    unsigned            num_fields;

    ty = Z3_get_type(z3_ctx, t);

    if (Z3_get_type_kind(z3_ctx, ty) != Z3_TUPLE_TYPE) {
      throw new z3_convt::conv_error("argument must be a tuple", irept());
    }

    num_fields = Z3_get_tuple_type_num_fields(z3_ctx, ty);

    if (i >= num_fields) {
      throw new z3_convt::conv_error("invalid tuple select, index is too big",
                                     irept());
    }

    Z3_const_decl_ast proj_decl = Z3_get_tuple_type_field_decl(z3_ctx, ty, i);
    return mk_unary_app(proj_decl, t);
}
