#ifndef _ESBMC_SOLVERS_SMT_ARRAY_SMT_CONV_H_
#define _ESBMC_SOLVERS_SMT_ARRAY_SMT_CONV_H_

// Something to abstract the flattening of arrays to QF_BV.

#include "smt_conv.h"

static inline bool
is_unbounded_array(const smt_sort *s)
{
  if (s->id != SMT_SORT_ARRAY)
    return false;

  if (s->get_domain_width() > 10)
    // This is either really large, or unbounded thus leading to a machine_int
    // sized domain. Either way, not a normal one.
    return true;
  else
    return false;
}

class array_ast : public smt_ast {
public:
#define array_downcast(x) static_cast<const array_ast*>(x)

  array_ast(const smt_sort *_s)
    : smt_ast(_s), symname(""), array_fields()
  {
  }

  array_ast(const smt_sort *_s,
                    const std::vector<const smt_ast *> &_a)
    : smt_ast(_s), symname(""), array_fields(_a)
  {
  }

  virtual ~array_ast(void) { }

  std::string symname; // Only if this was produced from mk_smt_symbol.

  std::vector<const smt_ast *> array_fields;
  unsigned int base_array_id;
  unsigned int array_update_num;
};


template <class subclass>
class array_convt : public subclass
{
public:
  struct array_select;
  struct array_with;

  array_convt(bool enable_cache, bool int_encoding, const namespacet &_ns,
              bool is_cpp, bool tuple_support, bool bools_in_arrs,
              bool can_init_inf_arrs);
  ~array_convt();

  // Things that the user must implement:
  virtual void assign_array_symbol(const std::string &name,
                                   const smt_ast *val) = 0;
  // And the get_bv and get_bool methods.

  // The api parts that this implements for smt_convt:

  virtual const smt_ast *convert_array_equality(const expr2tc &a,
                                                const expr2tc &b);
  virtual const smt_ast *mk_select(const expr2tc &array, const expr2tc &idx,
                                   const smt_sort *ressort);
  virtual const smt_ast *mk_store(const expr2tc &array, const expr2tc &idx,
                                  const expr2tc &value,
                                  const smt_sort *ressort);
  virtual const smt_ast *convert_array_of(const expr2tc &init_val,
                                          unsigned long domain_width);

  // The effective api for the solver-end to be using. Note that it has to
  //   a) Catch creation of array symbols and use fresh_array
  //   b) Catch array ITE's and call array_ite
  //   c) Pass control of 'get'ed array exprs to array_get
  //   d) Call add_array_constraints before calling dec_solve. Can be called
  //      multiple times.

  virtual smt_ast *fresh_array(const smt_sort *ms,
                               const std::string &name);
  smt_ast *array_ite(const smt_ast *cond,
                                   const smt_ast *true_arr,
                                   const smt_ast *false_arr,
                                   const smt_sort *thesort);
  expr2tc array_get(const smt_ast *a, const type2tc &subtype);
  void add_array_constraints(void);

  // Internal funk:

  const smt_ast *mk_unbounded_select(const array_ast *array,
                                     const expr2tc &idx,
                                     const smt_sort *ressort);
  const smt_ast *mk_unbounded_store(const array_ast *array,
                                    const expr2tc &idx,
                                    const smt_ast *value,
                                    const smt_sort *ressort);
  smt_ast *unbounded_array_ite(const array_ast *cond,
                                       const array_ast *true_arr,
                                       const array_ast *false_arr,
                                       const smt_sort *thesort);
  expr2tc fixed_array_get(const smt_ast *a, const type2tc &subtype);

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
    smt_ast *val;
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
        const smt_ast *val;
      } w;
      struct {
        unsigned int src_array_update_true;
        unsigned int src_array_update_false;
        const smt_ast *cond;
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
};

// And because this is a template...
#include "array_conv.cpp"

#endif /* _ESBMC_SOLVERS_SMT_ARRAY_SMT_CONV_H_ */

