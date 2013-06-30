#ifndef _ESBMC_SOLVERS_METASMT_MESTASMT_CONV_H_
#define _ESBMC_SOLVERS_METASMT_MESTASMT_CONV_H_

#include <solvers/smt/smt_conv.h>

#include <metaSMT/DirectSolver_Context.hpp>
#include <metaSMT/frontend/Logic.hpp>
#include <metaSMT/API/Assertion.hpp>
#include <metaSMT/backend/Z3_Backend.hpp>

// Crazyness: the desired solver is selected by some template meta programming.
// Therefore we have to statically know what solver we want to be using. To do
// that, we use the following war-def
typedef metaSMT::solver::Z3_Backend solvertype;
// Which defines our solvertype as being a Z3 solver.

typedef metaSMT::solver::Z3_Backend::result_type result_type;

namespace predtags = metaSMT::logic::tag;
namespace bvtags = metaSMT::logic::QF_BV::tag;
namespace arraytags = metaSMT::logic::Array::tag;

class metasmt_smt_sort : public smt_sort {
public:
#define metasmt_sort_downcast(x) static_cast<const metasmt_smt_sort*>(x)
  // Only three kinds of sorts supported: bools, bv's and arrays. Only
  // additional desireable information is the width.
  metasmt_smt_sort(smt_sort_kind i, unsigned int _width = 0)
    : smt_sort(i), width(_width)
  { }
  virtual ~metasmt_smt_sort() { }
  unsigned int width; // bv width
  unsigned int arrdom_width, arrrange_width; // arr sort widths
  virtual unsigned long get_domain_width(void) const {
    return arrdom_width;
  }
};

class metasmt_smt_ast : public smt_ast {
public:
#define metasmt_ast_downcast(x) static_cast<const metasmt_smt_ast*>(x)
  typedef std::list<std::pair<const smt_ast *, const smt_ast *> >
    unbounded_list_type;

  metasmt_smt_ast(const smt_sort *_s)
    : smt_ast(_s), restype(), symname(""), array_fields(), array_values()
  {
  }

  metasmt_smt_ast(result_type r, const smt_sort *_s)
    : smt_ast(_s), restype(r), symname(""), array_fields(), array_values()
  {
  }

  metasmt_smt_ast(result_type r, const smt_sort *_s, const std::string &s)
    : smt_ast(_s), restype(r), symname(s), array_fields(), array_values()
  {
  }

  metasmt_smt_ast(const smt_sort *_s, const std::vector<const smt_ast *> &a)
    : smt_ast(_s), restype(), symname(""), array_fields(a), array_values()
  {
  }

  metasmt_smt_ast(const smt_sort *_s, const unbounded_list_type &a)
    : smt_ast(_s), restype(), symname(""), array_fields(), array_values(a)
  {
  }

  virtual ~metasmt_smt_ast(void) { }

  bool is_unbounded_array(void) {
    if (sort->id != SMT_SORT_ARRAY)
      return false;

    if (sort->get_domain_width() > 10)
      // This is either really large, or unbounded thus leading to a machine_int
      // sized domain. Either way, not a normal one.
      return true;
    else
      return false;
  }


  result_type restype;
  std::string symname; // Only if this was produced from mk_smt_symbol.

  // If an array type, contains the set of array values. Identified by their
  // indexes.
  std::vector<const smt_ast *> array_fields;

  // Alternately, for unbounded arrays, what we want is a list of historical
  // assignments, and their corresponding values.
  unbounded_list_type array_values;
};

// copy+paste directly from the metaSMT documentation:
struct Lookup {
  typedef std::unordered_map<unsigned, std::string, std::hash<unsigned> >symmap;
  typedef std::unordered_map<std::string, metasmt_smt_ast*,
                             std::hash<std::string> > astmap;
  symmap &map_;
  astmap &astmap_;

  Lookup(symmap &map, astmap &a)
    : map_(map), astmap_(a) {}

  std::string operator()(unsigned id) {
    return map_[id];
  }

  metasmt_smt_ast *operator()(const std::string &str) {
    astmap::const_iterator it = astmap_.find(str);
    if (it == astmap_.end())
      return NULL;

    return it->second;
  }

  void insert(metasmt_smt_ast *ast, unsigned int id, std::string const &name) {
    map_.insert(std::make_pair(id, name));
    astmap_.insert(std::make_pair(name, ast));
  }
};

class metasmt_convt : public smt_convt
{
public:
  metasmt_convt(bool int_encoding, bool is_cpp, const namespacet &ns);
  virtual ~metasmt_convt();

  virtual void set_to(const expr2tc &expr, bool value);
  virtual resultt dec_solve();

  virtual expr2tc get(const expr2tc &expr);

  virtual tvt l_get(literalt a);
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

  virtual const smt_ast *mk_select(const expr2tc &array, const expr2tc &idx,
                                   const smt_sort *ressort);
  virtual const smt_ast *mk_store(const expr2tc &array, const expr2tc &idx,
                                  const expr2tc &value,
                                  const smt_sort *ressort);

  const smt_ast *mk_unbounded_select(const metasmt_smt_ast *array,
                                     const expr2tc &idx,
                                     const smt_sort *ressort);
  const smt_ast *mk_unbounded_store(const metasmt_smt_ast *array,
                                    const expr2tc &idx,
                                    const expr2tc &value,
                                    const smt_sort *ressort);

  const metasmt_smt_ast *fresh_array(const metasmt_smt_sort *ms,
                                     const std::string &name);
  const metasmt_smt_ast *array_ite(const metasmt_smt_ast *cond,
                                   const metasmt_smt_ast *true_arr,
                                   const metasmt_smt_ast *false_arr,
                                   const metasmt_smt_sort *thesort);
  const metasmt_smt_ast *unbounded_array_ite(const metasmt_smt_ast *cond,
                                             const metasmt_smt_ast *true_arr,
                                             const metasmt_smt_ast *false_arr,
                                             const metasmt_smt_sort *thesort);


  // Members
  solvertype ctx;
  Lookup::symmap symbols;
  Lookup::astmap astsyms;
  Lookup sym_lookup;
  bool bitblast_arrays;
};

#endif
