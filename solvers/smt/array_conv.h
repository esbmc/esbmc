#ifndef _ESBMC_SOLVERS_SMT_ARRAY_SMT_CONV_H_
#define _ESBMC_SOLVERS_SMT_ARRAY_SMT_CONV_H_

// Something to abstract the flattening of arrays to QF_BV.

#include <set>

#include <irep2.h>
#include "smt_conv.h"

static inline bool
is_unbounded_array(const smt_sort *s)
{
  if (s->id != SMT_SORT_ARRAY)
    return false;

  if (s->domain_width > 10)
    // This is either really large, or unbounded thus leading to a machine_int
    // sized domain. Either way, not a normal one.
    return true;
  else
    return false;
}

class array_convt;

class array_ast : public smt_ast {
public:
#define array_downcast(x) static_cast<const array_ast*>(x)

  array_ast(array_convt *actx, smt_convt *ctx, const smt_sort *_s)
    : smt_ast(ctx, _s), symname(""), array_fields(), array_ctx(actx)
  {
  }

  array_ast(array_convt *actx, smt_convt *ctx, const smt_sort *_s,
                    const std::vector<const smt_ast *> &_a)
    : smt_ast(ctx, _s), symname(""), array_fields(_a), array_ctx(actx)
  {
  }

  virtual ~array_ast(void) { }

  virtual smt_astt eq(smt_convt *ctx, smt_astt other) const;
  virtual smt_astt ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const;
  virtual void assign(smt_convt *ctx, smt_astt sym) const;
  virtual smt_astt update(smt_convt *ctx, smt_astt value,
                                unsigned int idx,
                                expr2tc idx_expr = expr2tc()) const;
  virtual smt_astt select(smt_convt *ctx, const expr2tc &idx) const;

  std::string symname; // Only if this was produced from mk_smt_symbol.

  std::vector<const smt_ast *> array_fields;
  unsigned int base_array_id;
  unsigned int array_update_num;

  array_convt *array_ctx;
};

class array_convt : public array_iface
{
public:
  struct array_select;
  struct array_with;

  array_convt(smt_convt *_ctx);
  ~array_convt();

  void convert_array_assign(const array_ast *src, smt_astt sym);
  const smt_ast *mk_select(const array_ast *array, const expr2tc &idx,
                                   const smt_sort *ressort);
  virtual smt_astt mk_store(const array_ast *array,
                                  const expr2tc &idx,
                                  smt_astt value,
                                  const smt_sort *ressort);
  virtual const smt_ast *convert_array_of(const expr2tc &init_val,
                                          unsigned long domain_width);

  smt_ast *mk_array_symbol(const std::string &name, const smt_sort *ms);
  smt_astt array_ite(const smt_ast *cond,
                                   const array_ast *true_arr,
                                   const array_ast *false_arr,
                                   const smt_sort *thesort);
  expr2tc get_array_elem(const smt_ast *a, uint64_t index,
                         const type2tc &subtype);
  void add_array_constraints_for_solving(void);

  // Internal funk:

  const smt_ast *mk_unbounded_select(const array_ast *array,
                                     const expr2tc &idx,
                                     const smt_sort *ressort);
  const smt_ast *mk_unbounded_store(const array_ast *array,
                                    const expr2tc &idx,
                                    const smt_ast *value,
                                    const smt_sort *ressort);
  smt_astt unbounded_array_ite(smt_astt cond,
                                       const array_ast *true_arr,
                                       const array_ast *false_arr,
                                       const smt_sort *thesort);

  // Array constraint beating

  void add_array_constraints(unsigned int arr);
  void execute_array_trans(std::vector<std::vector<const smt_ast *> > &data,
                           unsigned int arr,
                           unsigned int idx,
                           const std::map<expr2tc, unsigned> &idx_map,
                           const smt_sort *subtype);
  void collate_array_values(std::vector<const smt_ast *> &vals,
                            const std::map<expr2tc, unsigned> &idx_map,
                            const std::list<struct array_select> &idxs,
                            const smt_sort *subtype,
                            const smt_ast *init_val = NULL);
  void add_initial_ackerman_constraints(
                                    const std::vector<const smt_ast *> &vals,
                                    const std::map<expr2tc,unsigned> &idx_map);

  inline array_ast *new_ast(smt_sortt _s) {
    return new array_ast(this, ctx, _s);
  }

  inline array_ast *new_ast(smt_sortt _s,
      const std::vector<const smt_ast *> &_a) {
    return new array_ast(this, ctx, _s, _a);
  }

  // Members

  // Array tracking: each new root array (from fresh_array) gets its own
  // ID number which is stored. Then, whenever any operation occurs on it,
  // we add the index to the set contained in the following object. This
  // obtains all the tracking data required for CBMC-like array
  // bitblasting.
  std::vector<std::set<expr2tc> > array_indexes;

  // Self explanatory. Contains bitwidth of subtypes
  std::vector<unsigned int> array_subtypes;

  // Array /value/ tracking. For each array (outer vector) we have an inner
  // vector, each element of which corresponds to each 'with' operation
  // on an array. Within that is a list of indexes and free value'd
  // elements: whenever we select an element from an array, we return a
  // free value, and record it here. Assertions made later will link this
  // up with real values.
  struct array_select {
    unsigned int src_array_update_num;
    expr2tc idx;
    smt_astt val;
  };
  std::vector<std::vector<std::list<struct array_select> > > array_values;

  // Update records: For each array, for each 'with' operation, we record
  // the index used and the AST representation of the value assigned. We
  // also store the ID number of the source array, because due to phi's
  // we may branch arrays before joining.
  // We also record ite's here, because they're effectively an array update.
  // The is_ite boolean indicates whether the following union is a with or
  // an ite repr. If it's an ite, the two integers represent which two
  // historical update indexes of the array are operands of the ite.
  struct array_with {
    bool is_ite;
    expr2tc idx;
    union {
      struct {
        unsigned int src_array_update_num;
        smt_astt val;
      } w;
      struct {
        unsigned int src_array_update_true;
        unsigned int src_array_update_false;
        smt_astt cond;
      } i;
    } u;
  };
  std::vector<std::vector<struct array_with> > array_updates;

  // Map between base array identifiers and the value to initialize it with.
  // Only applicable to unbounded arrays.
  std::map<unsigned, const smt_ast *> array_of_vals;

  // Finally, for model building, we need all the past array values. Three
  // vectors, dimensions are arrays id's, historical point, array element,
  // respectively.
  std::vector<std::vector<std::vector<const smt_ast *> > > array_valuation;

  smt_convt *ctx;
};

#endif /* _ESBMC_SOLVERS_SMT_ARRAY_SMT_CONV_H_ */

