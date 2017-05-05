#ifndef _ESBMC_SOLVERS_SMTLIB_SMTLIB_CONV_H
#define _ESBMC_SOLVERS_SMTLIB_SMTLIB_CONV_H

#include <list>
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_tuple.h>
#include <solvers/smt/smt_tuple_flat.h>
#include <string>
#include <unistd.h>
#include <util/irep2.h>

class sexpr {
public:
  sexpr() : token(0), sexpr_list(), data() { sexpr_list.clear(); }
  unsigned int token; // If zero, then an sexpr list
  std::list<sexpr> sexpr_list;
  std::string data; // Text rep of parsed token.
};

class smtlib_smt_sort : public smt_sort {
public:
  explicit smtlib_smt_sort(smt_sort_kind k, unsigned int w)
    : smt_sort(k, w) { };
  explicit smtlib_smt_sort(smt_sort_kind k)
    : smt_sort(k) { }
  explicit smtlib_smt_sort(smt_sort_kind k, const smtlib_smt_sort *dom,
                  const smtlib_smt_sort *rag)
    : smt_sort(k, rag->data_width, dom->data_width), domain(dom), range(rag) { }

  const smtlib_smt_sort *domain;
  const smtlib_smt_sort *range;
};

class smtlib_smt_ast : public smt_ast {
public:
  smtlib_smt_ast(smt_convt *ctx, const smt_sort *s, smt_func_kind k)
    : smt_ast(ctx, s), kind(k) { }
  ~smtlib_smt_ast() { }
  virtual void dump() const { abort(); }

  smt_func_kind kind;
  std::string symname;
  BigInt intval;
  std::string realval;
  bool boolval;
  int extract_high;
  int extract_low;
  unsigned int num_args;
  const smt_ast *args[4];
};

class smtlib_convt : public smt_convt, public array_iface {
public:
  smtlib_convt(bool int_encoding, const namespacet &_ns,
               const optionst &_options);
  ~smtlib_convt();

  virtual resultt dec_solve();
  virtual tvt l_get(const smt_ast *a);
  virtual const std::string solver_text();

  virtual void assert_ast(const smt_ast *a);
  virtual smt_ast *mk_func_app(const smt_sort *s, smt_func_kind k,
                               const smt_ast * const *args,
                               unsigned int numargs);
  virtual smt_sort *mk_sort(const smt_sort_kind k, ...);
  virtual smt_ast *mk_smt_int(const mp_integer &theint, bool sign);
  virtual smt_ast *mk_smt_real(const std::string &str);
  virtual smt_ast *mk_smt_bvint(const mp_integer &theint, bool sign,
                                unsigned int w);
  virtual smt_ast *mk_smt_bvfloat(const ieee_floatt &thereal,
                                  unsigned ew, unsigned sw);
  virtual smt_astt mk_smt_bvfloat_nan(unsigned ew, unsigned sw);
  virtual smt_astt mk_smt_bvfloat_inf(bool sgn, unsigned ew, unsigned sw);
  virtual smt_astt mk_smt_bvfloat_rm(ieee_floatt::rounding_modet rm);
  virtual smt_astt mk_smt_typecast_from_bvfloat(const typecast2t &cast);
  virtual smt_astt mk_smt_typecast_to_bvfloat(const typecast2t &cast);
  virtual smt_astt mk_smt_nearbyint_from_float(const nearbyint2t &expr);
  virtual smt_astt mk_smt_bvfloat_arith_ops(const expr2tc &expr);
  virtual smt_ast *mk_smt_bool(bool val);
  virtual smt_ast *mk_smt_symbol(const std::string &name, const smt_sort *s);
  virtual smt_ast *mk_array_symbol(const std::string &name, const smt_sort *s,
                                   smt_sortt array_subtype);
  virtual smt_sort *mk_struct_sort(const type2tc &type);
  virtual smt_ast *mk_extract(const smt_ast *a, unsigned int high,
                              unsigned int low, const smt_sort *s);

  virtual void add_array_constraints_for_solving();

  const smt_ast *convert_array_of(smt_astt init_val,
                                  unsigned long domain_width);
  void push_array_ctx(void);
  void pop_array_ctx(void);

  virtual expr2tc get_bool(const smt_ast *a);
  virtual expr2tc get_bv(const type2tc &t, const smt_ast *a);
  virtual expr2tc get_array_elem(const smt_ast *array, uint64_t index,
                                 const type2tc &type);

  std::string sort_to_string(const smt_sort *s) const;
  unsigned int emit_terminal_ast(const smtlib_smt_ast *a, std::string &output);
  unsigned int emit_ast(const smtlib_smt_ast *ast, std::string &output);

  void push_ctx();
  void pop_ctx();

  // Members
  const optionst &options;
  pid_t solver_proc_pid;
  FILE *out_stream;
  FILE *in_stream;
  std::string solver_name;
  std::string solver_version;

  // Actual solving data
  // The set of symbols and their sorts.

  struct symbol_table_rec {
    std::string ident;
    unsigned int level;
    smt_sortt sort;
  };

  typedef boost::multi_index_container<
    symbol_table_rec,
    boost::multi_index::indexed_by<
      boost::multi_index::hashed_unique<
        BOOST_MULTI_INDEX_MEMBER(symbol_table_rec, std::string, ident)
      >,
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(symbol_table_rec, unsigned int, level),
        std::greater<unsigned int>
      >
    >
  > symbol_tablet;

  symbol_tablet symbol_table;
  std::vector<unsigned long> temp_sym_count;
  static const std::string temp_prefix;
};

#endif /* _ESBMC_SOLVERS_SMTLIB_SMTLIB_CONV_H */
