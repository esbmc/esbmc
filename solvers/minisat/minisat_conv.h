#ifndef _ESBMC_SOLVERS_SMTLIB_CONV_H_
#define _ESBMC_SOLVERS_SMTLIB_CONV_H_

#include <limits.h>
// For the sake of...
#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS
#include <stdint.h>
#include <inttypes.h>

#include <solvers/smt/smt_conv.h>

#include <core/Solver.h>

typedef Minisat::Lit Lit;
typedef Minisat::lbool lbool;
typedef std::vector<literalt> bvt; // sadface.jpg

class minisat_smt_sort : public smt_sort {
  // Record all the things.
  public:
#define minisat_sort_downcast(x) static_cast<const minisat_smt_sort*>(x)

  minisat_smt_sort(smt_sort_kind i)
    : smt_sort(i), width(0), sign(false), arrdom_width(0), arrrange_width(0)
  { }

  minisat_smt_sort(smt_sort_kind i, unsigned int _width, bool _sign)
    : smt_sort(i), width(_width), sign(_sign), arrdom_width(0),
      arrrange_width(0)
  { }

  minisat_smt_sort(smt_sort_kind i, unsigned int arrwidth,
                   unsigned int rangewidth)
    : smt_sort(i), width(0), sign(false), arrdom_width(arrwidth),
      arrrange_width(rangewidth)
  { }

  virtual ~minisat_smt_sort() { }
  unsigned int width; // bv width
  bool sign;
  unsigned int arrdom_width, arrrange_width; // arr sort widths

  virtual unsigned long get_domain_width(void) const {
    return arrdom_width;
  }

  bool is_unbounded_array(void) {
    if (id != SMT_SORT_ARRAY)
      return false;

    if (get_domain_width() > 10)
      // This is either really large, or unbounded thus leading to a machine_int
      // sized domain. Either way, not a normal one.
      return true;
    else
      return false;
  }
};

class minisat_smt_ast : public smt_ast {
public:
#define minisat_ast_downcast(x) static_cast<const minisat_smt_ast*>(x)
  minisat_smt_ast(const smt_sort *s) : smt_ast(s) { }

  // Everything is, to a greater or lesser extend, a vector of booleans, which
  // we'll represent as minisat Lit's.
  bvt bv;
};

class minisat_array_ast : public smt_ast {
public:
#define minisat_array_downcast(x) static_cast<const minisat_array_ast*>(x)

  minisat_array_ast(const smt_sort *_s)
    : smt_ast(_s), symname(""), array_fields()
  {
  }

  minisat_array_ast(const smt_sort *_s,
                    const std::vector<const smt_ast *> &_a)
    : smt_ast(_s), symname(""), array_fields(_a)
  {
  }

  virtual ~minisat_array_ast(void) { }

  bool is_unbounded_array(void) {
    return minisat_sort_downcast(sort)->is_unbounded_array();
  }

  std::string symname; // Only if this was produced from mk_smt_symbol.

  std::vector<const smt_ast *> array_fields;
  unsigned int base_array_id;
  unsigned int array_update_num;
};

class minisat_convt : public smt_convt {
public:
  struct array_select;
  struct array_with;
  typedef hash_map_cont<std::string, const smt_ast *, std::hash<std::string> > symtable_type;

  minisat_convt(bool int_encoding, const namespacet &_ns, bool is_cpp,
                const optionst &opts);
  ~minisat_convt();

  virtual resultt dec_solve();
  virtual expr2tc get(const expr2tc &expr);
  virtual const std::string solver_text();
  virtual tvt l_get(literalt l);
  virtual void assert_lit(const literalt &l);
  virtual smt_ast* mk_func_app(const smt_sort *ressort, smt_func_kind f,
                               const smt_ast* const* args, unsigned int num);
  virtual smt_sort* mk_sort(smt_sort_kind k, ...);
  virtual literalt mk_lit(const smt_ast *val);
  virtual smt_ast* mk_smt_int(const mp_integer &intval, bool sign);
  virtual smt_ast* mk_smt_real(const std::string &value);
  virtual smt_ast* mk_smt_bvint(const mp_integer &inval, bool sign,
                                unsigned int w);
  virtual smt_ast* mk_smt_bool(bool boolval);
  virtual smt_ast* mk_smt_symbol(const std::string &name, const smt_sort *sort);
  virtual smt_sort* mk_struct_sort(const type2tc &t);
  virtual smt_sort* mk_union_sort(const type2tc&t);
  virtual smt_ast* mk_extract(const smt_ast *src, unsigned int high,
                              unsigned int low, const smt_sort *s);

  virtual literalt new_variable();
  virtual const smt_ast *convert_array_equality(const expr2tc &a,
                                                const expr2tc &b);

  virtual const smt_ast *lit_to_ast(const literalt &l);

  minisat_smt_ast *mk_ast_equality(const minisat_smt_ast *a,
                                   const minisat_smt_ast *b,
                                   const smt_sort *ressort);

  // Things imported from CBMC, more or less
  void convert(const bvt &bv, Minisat::vec<Lit> &dest);
  void eliminate_duplicates(const bvt &bv, bvt &dest);
  literalt lnot(literalt a);
  literalt lselect(literalt a, literalt b, literalt c);
  literalt lequal(literalt a, literalt b);
  literalt limplies(literalt a, literalt b);
  literalt lxor(literalt a, literalt b);
  literalt lor(literalt a, literalt b);
  literalt land(literalt a, literalt b);
  literalt land(const bvt &bv);
  void bvand(const bvt &bv0, const bvt &bv1, bvt &output);
  literalt lor(const bvt &bv);
  void gate_xor(literalt a, literalt b, literalt o);
  void gate_or(literalt a, literalt b, literalt o);
  void gate_and(literalt a, literalt b, literalt o);
  void set_equal(literalt a, literalt b);
  void setto(literalt a, bool val);
  virtual void lcnf(const bvt &bv);

  void full_adder(const bvt &op0, const bvt &op1, bvt &output,
                  literalt carry_in, literalt &carry_out);
  literalt carry(literalt a, literalt b, literalt c);
  literalt carry_out(const bvt &a, const bvt &b, literalt c);
  literalt equal(const bvt &op0, const bvt &op1);
  literalt lt_or_le(bool or_equal, const bvt &bv0, const bvt &bv1,
                    bool is_signed);
  void invert(bvt &bv);

  literalt unsigned_less_than(const bvt &arg0, const bvt &arg1);
  void unsigned_multiplier(const bvt &op0, const bvt &bv1, bvt &output);
  void signed_multiplier(const bvt &op0, const bvt &bv1, bvt &output);
  void cond_negate(const bvt &vals, bvt &out, literalt cond);
  void negate(const bvt &inp, bvt &oup);
  void incrementer(const bvt &inp, const literalt &carryin, literalt carryout,
                   bvt &oup);

  expr2tc get_bool(const smt_ast *a);
  expr2tc get_bv(const type2tc &t, const smt_ast *a);

  virtual const smt_ast *mk_select(const expr2tc &array, const expr2tc &idx,
                                   const smt_sort *ressort);
  virtual const smt_ast *mk_store(const expr2tc &array, const expr2tc &idx,
                                  const expr2tc &value,
                                  const smt_sort *ressort);

  virtual const smt_ast *convert_array_of(const expr2tc &init_val,
                                          unsigned long domain_width);

  const smt_ast *mk_unbounded_select(const minisat_array_ast *array,
                                     const expr2tc &idx,
                                     const smt_sort *ressort);
  const smt_ast *mk_unbounded_store(const minisat_array_ast *array,
                                    const expr2tc &idx,
                                    const smt_ast *value,
                                    const smt_sort *ressort);

  const smt_ast *fresh_array(const minisat_smt_sort *ms,
                             const std::string &name);
  const minisat_array_ast *array_ite(const minisat_smt_ast *cond,
                                   const minisat_array_ast *true_arr,
                                   const minisat_array_ast *false_arr,
                                   const minisat_smt_sort *thesort);
  const minisat_array_ast *unbounded_array_ite(const minisat_smt_ast *cond,
                                       const minisat_array_ast *true_arr,
                                       const minisat_array_ast *false_arr,
                                       const minisat_smt_sort *thesort);

  expr2tc fixed_array_get(const smt_ast *a, const type2tc &subtype);
  expr2tc array_get(const smt_ast *a, const type2tc &subtype);
  void add_array_constraints(void);
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
                                    const std::map<expr2tc,unsigned> &idx_map,
                                    const smt_sort *subtype);

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

  Minisat::Solver solver;
  const optionst &options;
  symtable_type sym_table;
};

#endif /* _ESBMC_SOLVERS_SMTLIB_CONV_H_ */
