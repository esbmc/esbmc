#ifndef _ESBMC_SOLVERS_SMT_SMT_TUPLE_H_
#define _ESBMC_SOLVERS_SMT_SMT_TUPLE_H_

#include "smt_conv.h"

// Abstract class defining the interface required for creating tuples.
class tuple_iface {
  /** Create a sort representing a struct. i.e., a tuple. Ideally this should
   *  actually be part of the overridden tuple api, but due to history it isn't
   *  yet. If solvers don't support tuples, implement this to abort.
   *  @param type The struct type to create a tuple representation of.
   *  @return The tuple representation of the type, wrapped in an smt_sort. */
  virtual smt_sortt mk_struct_sort(const type2tc &type) = 0;

  /** Create a sort representing a union. i.e., a tuple. Ideally this should
   *  actually be part of the overridden tuple api, but due to history it isn't
   *  yet. If solvers don't support tuples, implement this to abort.
   *  @param type The union type to create a tuple representation of.
   *  @return The tuple representation of the type, wrapped in an smt_sort. */
  virtual smt_sortt mk_union_sort(const type2tc &type) = 0;

  /** Create a new tuple from a struct definition.
   *  @param structdef A constant_struct2tc, describing all the members of the
   *         tuple to create.
   *  @return AST representing the created tuple */
  virtual smt_astt tuple_create(const expr2tc &structdef) = 0;

  virtual smt_astt union_create(const expr2tc &unidef) = 0;

  /** Create a fresh tuple, with freely valued fields.
   *  @param s Sort of the tuple to create
   *  @return AST representing the created tuple */
  virtual smt_astt tuple_fresh(smt_sortt s, std::string name = "") = 0;

  /** Create an array of tuple values. Takes a type, and an array of ast's,
   *  and creates an array where the elements have the value of the input asts.
   *  Essentially a way of converting a constant_array2tc, with tuple type.
   *  @param array_type Type of the array we will be creating, with size.
   *  @param input_args Array of ASTs to form the elements of this array. Must
   *         have the size indicated by array_type. (This method can't be
   *         used to create nondeterministically or infinitely sized arrays).
   *  @param const_array If true, only the first element of input_args is valid,
   *         and is repeated for every element in this (fixed size) array.
   *  @param domain Sort of the domain of this array. */
  virtual smt_astt tuple_array_create(const type2tc &array_type,
                                            smt_astt *input_args,
                                            bool const_array,
                                            smt_sortt domain) = 0;

  /** Create a potentially /large/ array of tuples. This is called when we
   *  encounter an array_of operation, with a very large array size, of tuple
   *  sort.
   *  @param Expression of tuple value to populate this array with.
   *  @param domain_width The size of array to create, in domain bits.
   *  @return An AST representing an array of the tuple value, init_value. */
  virtual smt_astt tuple_array_of(const expr2tc &init_value,
                                        unsigned long domain_width) = 0;

  /** Convert a symbol2tc to a tuple_smt_ast */
  virtual smt_astt mk_tuple_symbol(const expr2tc &expr) = 0;

  /** Like mk_tuple_symbol, but for arrays */
  virtual smt_astt mk_tuple_array_symbol(const expr2tc &expr) = 0;

  /** Extract the assignment to a tuple-typed symbol from the SMT solvers
   *  model */
  virtual expr2tc tuple_get(const expr2tc &expr) = 0;

  /** Extract the assignment to a tuple-array symbol from the SMT solvers
   *  model */
  virtual expr2tc tuple_array_get(const expr2tc &expr) = 0;
};

// While we're at it, define the default implementation details.

/** Function app representing a tuple sorted value.
 *  This AST represents any kind of SMT function that results in something of
 *  a tuple sort. As documented in smt_tuple.c, the result of any kind of
 *  tuple operation that gets flattened is a symbol prefix, which is what this
 *  ast actually stores.
 *
 *  This AST should only be used in smt_tuple.c, if you're using it elsewhere
 *  think very hard about what you're trying to do. Its creation should also
 *  only occur if there is no tuple support in the solver being used, and a
 *  tuple creating method has been called.
 *
 *  @see smt_tuple.c */
class tuple_smt_ast : public smt_ast {
public:
  /** Primary constructor.
   *  @param s The sort of the tuple, of type tuple_smt_sort.
   *  @param _name The symbol prefix of the variables representing this tuples
   *               value. */
  tuple_smt_ast (smt_convt *ctx, smt_sortt s, const std::string &_name)
    : smt_ast(ctx, s), name(_name) { }
  virtual ~tuple_smt_ast() { }

  /** The symbol prefix of the variables representing this tuples value, as a
   *  string (i.e., no associated type). */
  const std::string name;

  std::vector<smt_astt> elements;

  virtual smt_astt ite(smt_convt *ctx, smt_astt cond,
      smt_astt falseop) const;
  virtual smt_astt eq(smt_convt *ctx, smt_astt other) const;
  virtual smt_astt assign(smt_convt *ctx, const expr2tc &sym) const;
  virtual smt_astt update(smt_convt *ctx, smt_astt value,
                                unsigned int idx,
                                expr2tc idx_expr = expr2tc()) const;
  virtual smt_astt select(smt_convt *ctx, const expr2tc &idx) const;
  virtual smt_astt project(smt_convt *ctx, unsigned int elem) const;

  void make_free(smt_convt *ctx);
};

inline tuple_smt_astt
to_tuple_ast(smt_astt a)
{
  tuple_smt_astt ta = dynamic_cast<tuple_smt_astt>(a);
  assert(ta != NULL && "Tuple AST mismatch");
  return ta;
}

inline tuple_smt_sortt
to_tuple_sort(smt_sortt a)
{
  tuple_smt_sortt ta = dynamic_cast<tuple_smt_sortt >(a);
  assert(ta != NULL && "Tuple AST mismatch");
  return ta;
}

class array_smt_ast : public tuple_smt_ast
{
public:
  array_smt_ast (smt_convt *ctx, smt_sortt s, const std::string &_name);
  virtual ~array_smt_ast() { }

  virtual smt_astt ite(smt_convt *ctx, smt_astt cond,
      smt_astt falseop) const;
  virtual smt_astt eq(smt_convt *ctx, smt_astt other) const;
  virtual smt_astt assign(smt_convt *ctx, const expr2tc &sym) const;
  virtual smt_astt update(smt_convt *ctx, smt_astt value,
                                unsigned int idx,
                                expr2tc idx_expr = expr2tc()) const;
  virtual smt_astt select(smt_convt *ctx, const expr2tc &idx) const;
  virtual smt_astt project(smt_convt *ctx, unsigned int elem) const;

  bool is_still_free;
};

inline array_smt_astt
to_array_ast(smt_astt a)
{
  array_smt_astt ta = dynamic_cast<array_smt_astt>(a);
  assert(ta != NULL && "Tuple-Array AST mismatch");
  return ta;
}

extern inline
array_smt_ast::array_smt_ast(smt_convt *ctx, smt_sortt s,
    const std::string &_name)
    : tuple_smt_ast(ctx, s, _name) {
  // A new array is inherently fresh; thus field each element slot with
  // a fresh new array.

  is_still_free = true;

  tuple_smt_sortt ts = to_tuple_sort(s);
  const array_type2t &array_type = to_array_type(ts->thetype);
  const struct_union_data &strct = ctx->get_type_def(array_type.subtype);

  unsigned int i = 0;
  elements.resize(strct.members.size());
  forall_types(it, strct.members) {
    type2tc new_arrtype(new array_type2t(*it, array_type.array_size,
                                         array_type.size_is_infinite));
    smt_sortt newsort = ctx->convert_sort(new_arrtype);

    // Normal elements are just normal arrays. Everything else requires
    // a recursive array_smt_ast.
    if (is_tuple_ast_type(*it)) {
      elements[i] = new array_smt_ast(ctx, newsort,
                              _name + "." + strct.member_names[i].as_string());
    } else {
      elements[i] = ctx->mk_fresh(newsort, "array_smt_ast");
    }

    i++;
  }
}

#endif /* _ESBMC_SOLVERS_SMT_SMT_TUPLE_H_ */
