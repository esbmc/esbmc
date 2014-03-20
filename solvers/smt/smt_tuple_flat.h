#include "smt_conv.h"
#include <namespace.h>

class tuple_node_smt_ast;
class array_node_smt_ast;
class tuple_sym_smt_ast;
class array_sym_smt_ast;
class tuple_smt_sort;
typedef const tuple_node_smt_ast *tuple_node_smt_astt;
typedef const array_node_smt_ast *array_node_smt_astt;
typedef const tuple_sym_smt_ast *tuple_sym_smt_astt;
typedef const array_sym_smt_ast *array_sym_smt_astt;
typedef const tuple_smt_sort *tuple_smt_sortt;

/** Storage for flattened tuple sorts.
 *  When flattening tuples (and arrays of them) down to SMT, we need to store
 *  additional type data. This sort is used in tuple code to record that data.
 *  @see smt_tuple.cpp */
class tuple_smt_sort : public smt_sort
{
public:
  /** Actual type (struct or array of structs) of the tuple that's been
   * flattened */
  const type2tc thetype;

  tuple_smt_sort(const type2tc &type)
    : smt_sort(SMT_SORT_STRUCT, 0, 0), thetype(type)
  {
  }

  tuple_smt_sort(const type2tc &type, unsigned long range_width,
                 unsigned long dom_width)
    : smt_sort(SMT_SORT_ARRAY, range_width, dom_width), thetype(type)
  {
  }

  virtual ~tuple_smt_sort() { }
};

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
class tuple_node_smt_ast : public smt_ast {
public:
  /** Primary constructor.
   *  @param s The sort of the tuple, of type tuple_smt_sort.
   *  @param _name The symbol prefix of the variables representing this tuples
   *               value. */
  tuple_node_smt_ast (smt_convt *ctx, smt_sortt s, const std::string &_name)
    : smt_ast(ctx, s), name(_name) { }
  virtual ~tuple_node_smt_ast() { }

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

inline tuple_node_smt_astt
to_tuple_node_ast(smt_astt a)
{
  tuple_node_smt_astt ta = dynamic_cast<tuple_node_smt_astt>(a);
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

class array_node_smt_ast : public tuple_node_smt_ast
{
public:
  array_node_smt_ast (smt_convt *ctx, smt_sortt s, const std::string &_name);
  virtual ~array_node_smt_ast() { }

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

inline array_node_smt_astt
to_array_node_ast(smt_astt a)
{
  array_node_smt_astt ta = dynamic_cast<array_node_smt_astt>(a);
  assert(ta != NULL && "Tuple-Array AST mismatch");
  return ta;
}

extern inline
array_node_smt_ast::array_node_smt_ast(smt_convt *ctx, smt_sortt s,
    const std::string &_name)
    : tuple_node_smt_ast(ctx, s, _name) {
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
    // a recursive array_node_smt_ast.
    if (is_tuple_ast_type(*it)) {
      elements[i] = new array_node_smt_ast(ctx, newsort,
                              _name + "." + strct.member_names[i].as_string());
    } else {
      elements[i] = ctx->mk_fresh(newsort, "array_node_smt_ast");
    }

    i++;
  }
}

class smt_tuple_node_flattener : public tuple_iface
{
public:
  smt_tuple_node_flattener(smt_convt *_ctx, const namespacet &_ns)
    : tuple_iface(), ctx(_ctx), ns(_ns) { }

  virtual smt_sortt mk_struct_sort(const type2tc &type);
  virtual smt_sortt mk_union_sort(const type2tc &type);
  virtual smt_astt tuple_create(const expr2tc &structdef);
  virtual smt_astt union_create(const expr2tc &unidef);
  virtual smt_astt tuple_fresh(smt_sortt s, std::string name = "");
  virtual smt_astt tuple_array_of(const expr2tc &init_value,
                                            unsigned long domain_width);
  virtual smt_astt mk_tuple_symbol(const expr2tc &expr);
  virtual smt_astt mk_tuple_array_symbol(const expr2tc &expr);
  virtual expr2tc tuple_get(const expr2tc &expr);

  expr2tc tuple_get_rec(tuple_node_smt_astt tuple);
  virtual smt_astt tuple_array_create(const type2tc &array_type,
                                            smt_astt *input_args,
                                            bool const_array,
                                            smt_sortt domain);

  smt_convt *ctx;
  const namespacet &ns;
};

class tuple_sym_smt_ast : public smt_ast {
public:
  /** Primary constructor.
   *  @param s The sort of the tuple, of type tuple_smt_sort.
   *  @param _name The symbol prefix of the variables representing this tuples
   *               value. */
  tuple_sym_smt_ast (smt_convt *ctx, smt_sortt s, const std::string &_name)
    : smt_ast(ctx, s), name(_name) { }
  virtual ~tuple_sym_smt_ast() { }

  /** The symbol prefix of the variables representing this tuples value, as a
   *  string (i.e., no associated type). */
  const std::string name;


  virtual smt_astt ite(smt_convt *ctx, smt_astt cond,
      smt_astt falseop) const;
  virtual smt_astt eq(smt_convt *ctx, smt_astt other) const;
  virtual smt_astt update(smt_convt *ctx, smt_astt value,
                                unsigned int idx,
                                expr2tc idx_expr = expr2tc()) const;
  virtual smt_astt select(smt_convt *ctx, const expr2tc &idx) const;
  virtual smt_astt project(smt_convt *ctx, unsigned int elem) const;
};

class array_sym_smt_ast : public tuple_sym_smt_ast
{
public:
  array_sym_smt_ast (smt_convt *ctx, smt_sortt s, const std::string &_name)
    : tuple_sym_smt_ast(ctx, s, _name) { }
  virtual ~array_sym_smt_ast() { }

  virtual smt_astt ite(smt_convt *ctx, smt_astt cond,
      smt_astt falseop) const;
  virtual smt_astt eq(smt_convt *ctx, smt_astt other) const;
  virtual smt_astt update(smt_convt *ctx, smt_astt value,
                                unsigned int idx,
                                expr2tc idx_expr = expr2tc()) const;
  virtual smt_astt select(smt_convt *ctx, const expr2tc &idx) const;
  virtual smt_astt project(smt_convt *ctx, unsigned int elem) const;
};

inline tuple_sym_smt_astt
to_tuple_sym_ast(smt_astt a)
{
  tuple_sym_smt_astt ta = dynamic_cast<tuple_sym_smt_astt>(a);
  assert(ta != NULL && "Tuple AST mismatch");
  return ta;
}

inline array_sym_smt_astt
to_array_sym_ast(smt_astt a)
{
  array_sym_smt_astt ta = dynamic_cast<array_sym_smt_astt>(a);
  assert(ta != NULL && "Tuple-Array AST mismatch");
  return ta;
}

class smt_tuple_sym_flattener : public tuple_iface
{
public:
  smt_tuple_sym_flattener(smt_convt *_ctx, const namespacet &_ns)
    : tuple_iface(), ctx(_ctx), ns(_ns) { }

  virtual smt_sortt mk_struct_sort(const type2tc &type);
  virtual smt_sortt mk_union_sort(const type2tc &type);
  virtual smt_astt tuple_create(const expr2tc &structdef);
  virtual smt_astt union_create(const expr2tc &unidef);
  virtual smt_astt tuple_fresh(smt_sortt s, std::string name = "");
  virtual smt_astt tuple_array_of(const expr2tc &init_value,
                                            unsigned long domain_width);
  virtual smt_astt mk_tuple_symbol(const expr2tc &expr);
  virtual smt_astt mk_tuple_array_symbol(const expr2tc &expr);
  virtual expr2tc tuple_get(const expr2tc &expr);

  expr2tc tuple_get_rec(tuple_node_smt_astt tuple);
  virtual smt_astt tuple_array_create(const type2tc &array_type,
                                            smt_astt *input_args,
                                            bool const_array,
                                            smt_sortt domain);

  smt_convt *ctx;
  const namespacet &ns;
};
