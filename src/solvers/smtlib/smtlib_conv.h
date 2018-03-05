#ifndef _ESBMC_SOLVERS_SMTLIB_SMTLIB_CONV_H
#define _ESBMC_SOLVERS_SMTLIB_SMTLIB_CONV_H

#include <list>
#include <solvers/smt/smt_conv.h>
#include <string>
#include <unistd.h>
#include <util/irep2.h>

class sexpr
{
public:
  sexpr() : token(0)
  {
    sexpr_list.clear();
  }
  unsigned int token; // If zero, then an sexpr list
  std::list<sexpr> sexpr_list;
  std::string data; // Text rep of parsed token.
};

class smtlib_smt_sort : public smt_sort
{
public:
  explicit smtlib_smt_sort(smt_sort_kind k, unsigned int w) : smt_sort(k, w){};
  explicit smtlib_smt_sort(smt_sort_kind k) : smt_sort(k)
  {
  }
  explicit smtlib_smt_sort(
    smt_sort_kind k,
    const smtlib_smt_sort *dom,
    const smtlib_smt_sort *rag)
    : smt_sort(k, dom->get_data_width(), rag->get_domain_width()),
      domain(dom),
      range(rag)
  {
  }

  const smtlib_smt_sort *domain;
  const smtlib_smt_sort *range;
};

class smtlib_smt_ast : public smt_ast
{
public:
  smtlib_smt_ast(smt_convt *ctx, const smt_sort *s, smt_func_kind k)
    : smt_ast(ctx, s), kind(k)
  {
  }
  ~smtlib_smt_ast() override = default;

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

class smtlib_convt : public smt_convt, public array_iface, public fp_convt
{
public:
  smtlib_convt(bool int_encoding, const namespacet &_ns);
  ~smtlib_convt() override;

  resultt dec_solve() override;
  const std::string solver_text() override;

  void assert_ast(const smt_ast *a) override;
  smt_ast *mk_func_app(
    const smt_sort *s,
    smt_func_kind k,
    const smt_ast *const *args,
    unsigned int numargs) override;

  smt_sortt mk_bool_sort() override;
  smt_sortt mk_real_sort() override;
  smt_sortt mk_int_sort() override;
  smt_sortt mk_bv_sort(const smt_sort_kind k, std::size_t width) override;
  smt_sortt mk_array_sort(smt_sortt domain, smt_sortt range) override;
  smt_sortt mk_bv_fp_sort(std::size_t ew, std::size_t sw) override;
  smt_sortt mk_bv_fp_rm_sort() override;

  smt_ast *mk_smt_int(const mp_integer &theint, bool sign) override;
  smt_ast *mk_smt_real(const std::string &str) override;
  smt_astt mk_smt_bv(smt_sortt s, const mp_integer &theint) override;
  smt_ast *mk_smt_bool(bool val) override;
  smt_ast *mk_smt_symbol(const std::string &name, const smt_sort *s) override;
  smt_ast *mk_array_symbol(
    const std::string &name,
    const smt_sort *s,
    smt_sortt array_subtype) override;
  smt_sort *mk_struct_sort(const type2tc &type);
  smt_astt
  mk_extract(const smt_ast *a, unsigned int high, unsigned int low) override;
  smt_astt mk_sign_ext(smt_astt a, unsigned int topwidth) override;
  smt_astt mk_zero_ext(smt_astt a, unsigned int topwidth) override;

  void add_array_constraints_for_solving() override;

  const smt_ast *
  convert_array_of(smt_astt init_val, unsigned long domain_width) override;
  void push_array_ctx() override;
  void pop_array_ctx() override;

  bool get_bool(const smt_ast *a) override;
  BigInt get_bv(smt_astt a) override;
  expr2tc get_array_elem(
    const smt_ast *array,
    uint64_t index,
    const type2tc &type) override;

  std::string sort_to_string(const smt_sort *s) const;
  unsigned int emit_terminal_ast(const smtlib_smt_ast *a, std::string &output);
  unsigned int emit_ast(const smtlib_smt_ast *ast, std::string &output);

  void push_ctx() override;
  void pop_ctx() override;

  // Members
  pid_t solver_proc_pid;
  FILE *out_stream;
  FILE *in_stream;
  std::string solver_name;
  std::string solver_version;

  // Actual solving data
  // The set of symbols and their sorts.

  struct symbol_table_rec
  {
    std::string ident;
    unsigned int level;
    smt_sortt sort;
  };

  typedef boost::multi_index_container<
    symbol_table_rec,
    boost::multi_index::indexed_by<
      boost::multi_index::hashed_unique<
        BOOST_MULTI_INDEX_MEMBER(symbol_table_rec, std::string, ident)>,
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(symbol_table_rec, unsigned int, level),
        std::greater<unsigned int>>>>
    symbol_tablet;

  symbol_tablet symbol_table;
  std::vector<unsigned long> temp_sym_count;
  static const std::string temp_prefix;
};

#endif /* _ESBMC_SOLVERS_SMTLIB_SMTLIB_CONV_H */
