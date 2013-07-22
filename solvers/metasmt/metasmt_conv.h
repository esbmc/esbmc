#ifndef _ESBMC_SOLVERS_METASMT_MESTASMT_CONV_H_
#define _ESBMC_SOLVERS_METASMT_MESTASMT_CONV_H_

#include <set>
#include <map>

#include <irep2.h>

#include <solvers/smt/smt_conv.h>

#include <metaSMT/frontend/Logic.hpp>
#include <metaSMT/frontend/Array.hpp>

// To save everyone's faces from being knawed off by killer weasels powered by
// GCC error messages and inflation, avoid turning metasmt_convt into a
// template. It'd be far too funky, seriously. Every time you did something
// wrong all the errors would trebble in length and you'd have to guess whether
// it's solver-specific or not.
//
// So instead, have the source and header for metasmt preprocessed into a
// source file specific to the solver; and have some preprocesesor defines
// define the name of the classes, and the type of the solver to be used.
// Yes, this is a preprocessor hack, but in the long run you will apreciate
// your continued existance and thank me for it.
//
// Definitions: SOLVER_PREFIX = what to jam on the front of "_convt" and all
// other paraphernalia to form class names.
// SOLVER_TYPE = the solver we're going to be using.

// That being said, I've completely failed to macro, so have some external
// definitions for the moment.

typedef SOLVER_TYPE solvertype;
//typedef metaSMT::solver::Z3_Backend solvertype;
// Which defines our solvertype as being a Z3 solver.

typedef SOLVER_TYPE::result_type result_type;
//typedef metaSMT::solver::Z3_Backend::result_type result_type;

namespace predtags = metaSMT::logic::tag;
namespace bvtags = metaSMT::logic::QF_BV::tag;
namespace arraytags = metaSMT::logic::Array::tag;

class metasmt_smt_sort : public smt_sort {
public:
#define metasmt_sort_downcast(x) static_cast<const metasmt_smt_sort*>(x)
  // Only three kinds of sorts supported: bools, bv's and arrays. Only
  // additional desireable information is the width.

  metasmt_smt_sort(smt_sort_kind i) : smt_sort(i) { }
  metasmt_smt_sort(smt_sort_kind i, unsigned long w) : smt_sort(i, w) { }
  metasmt_smt_sort(smt_sort_kind i, unsigned long d_w, unsigned long dom_w)
    : smt_sort(i, d_w, dom_w) { }

  virtual ~metasmt_smt_sort() { }

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

class metasmt_smt_ast : public smt_ast {
public:
#define metasmt_ast_downcast(x) static_cast<const metasmt_smt_ast*>(x)
  metasmt_smt_ast(const smt_sort *_s)
    : smt_ast(_s), restype(), symname("")
  {
  }

  metasmt_smt_ast(result_type r, const smt_sort *_s)
    : smt_ast(_s), restype(r), symname("")
  {
  }

  metasmt_smt_ast(result_type r, const smt_sort *_s, const std::string &s)
    : smt_ast(_s), restype(r), symname(s)
  {
  }

  virtual ~metasmt_smt_ast(void) { }

  result_type restype;
  std::string symname; // Only if this was produced from mk_smt_symbol.
};

class metasmt_array_ast : public smt_ast {
public:
#define metasmt_array_downcast(x) static_cast<const metasmt_array_ast*>(x)

  metasmt_array_ast(const smt_sort *_s)
    : smt_ast(_s), symname(""), array_fields()
  {
  }

  metasmt_array_ast(const smt_sort *_s,
                    const std::vector<const smt_ast *> &_a)
    : smt_ast(_s), symname(""), array_fields(_a)
  {
  }

  virtual ~metasmt_array_ast(void) { }

  bool is_unbounded_array(void) {
    return metasmt_sort_downcast(sort)->is_unbounded_array();
  }

  std::string symname; // Only if this was produced from mk_smt_symbol.

  std::vector<const smt_ast *> array_fields;
  unsigned int base_array_id;
  unsigned int array_update_num;
};

// copy+paste directly from the metaSMT documentation:
struct Lookup {
  typedef std::unordered_map<unsigned, std::string, std::hash<unsigned> >symmap;
  typedef std::unordered_map<std::string, smt_ast*,
                             std::hash<std::string> > astmap;
  symmap &map_;
  astmap &astmap_;

  Lookup(symmap &map, astmap &a)
    : map_(map), astmap_(a) {}

  std::string operator()(unsigned id) {
    return map_[id];
  }

  smt_ast *operator()(const std::string &str) {
    astmap::const_iterator it = astmap_.find(str);
    if (it == astmap_.end())
      return NULL;

    return it->second;
  }

  void insert(smt_ast *ast, unsigned int id, std::string const &name) {
    map_.insert(std::make_pair(id, name));
    astmap_.insert(std::make_pair(name, ast));
  }
};

class metasmt_convt : public smt_convt
{
public:
  struct array_select;
  struct array_with;

  metasmt_convt(bool int_encoding, bool is_cpp, const namespacet &ns);
  virtual ~metasmt_convt();

  virtual void set_to(const expr2tc &expr, bool value);
  virtual resultt dec_solve();

  virtual expr2tc get(const expr2tc &expr);

  virtual tvt l_get(literalt a);
  virtual tvt l_get(const smt_ast *a);
  virtual const std::string solver_text();

  virtual void assert_lit(const literalt &l);
  virtual smt_ast *mk_func_app(const smt_sort *s, smt_func_kind k,
                               const smt_ast * const *args,
                               unsigned int numargs);
  virtual smt_sort *mk_sort(const smt_sort_kind k, ...);
  virtual literalt mk_lit(const smt_ast *s);
  virtual smt_ast *mk_smt_int(const mp_integer &theint, bool sign);
  virtual smt_ast *mk_smt_real(const std::string &str);
  virtual smt_ast *mk_smt_bvint(const mp_integer &theint, bool sign,
                                unsigned int w);
  virtual smt_ast *mk_smt_bool(bool val);
  virtual smt_ast *mk_smt_symbol(const std::string &name, const smt_sort *s);
  virtual smt_sort *mk_struct_sort(const type2tc &type) ;
  virtual smt_sort *mk_union_sort(const type2tc &type);
  virtual smt_ast *mk_extract(const smt_ast *a, unsigned int high,
                              unsigned int low, const smt_sort *s);

#ifdef SOLVER_BITBLAST_ARRAYS
  virtual const smt_ast *mk_select(const expr2tc &array, const expr2tc &idx,
                                   const smt_sort *ressort);
  virtual const smt_ast *mk_store(const expr2tc &array, const expr2tc &idx,
                                  const expr2tc &value,
                                  const smt_sort *ressort);

  virtual const smt_ast *convert_array_of(const expr2tc &init_val,
                                          unsigned long domain_width);

  const smt_ast *mk_unbounded_select(const metasmt_array_ast *array,
                                     const expr2tc &idx,
                                     const smt_sort *ressort);
  const smt_ast *mk_unbounded_store(const metasmt_array_ast *array,
                                    const expr2tc &idx,
                                    const smt_ast *value,
                                    const smt_sort *ressort);

  const smt_ast *fresh_array(const metasmt_smt_sort *ms,
                             const std::string &name);
  const metasmt_array_ast *array_ite(const metasmt_smt_ast *cond,
                                   const metasmt_array_ast *true_arr,
                                   const metasmt_array_ast *false_arr,
                                   const metasmt_smt_sort *thesort);
  const metasmt_array_ast *unbounded_array_ite(const metasmt_smt_ast *cond,
                                       const metasmt_array_ast *true_arr,
                                       const metasmt_array_ast *false_arr,
                                       const metasmt_smt_sort *thesort);

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

#endif /* SOLVER_BITBLAST_ARRAYS */

  // Members
  solvertype ctx;
  Lookup::symmap symbols;
  Lookup::astmap astsyms;
  Lookup sym_lookup;

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
};

#endif
