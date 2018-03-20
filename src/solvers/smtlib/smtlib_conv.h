#ifndef _ESBMC_SOLVERS_SMTLIB_SMTLIB_CONV_H
#define _ESBMC_SOLVERS_SMTLIB_SMTLIB_CONV_H

#include <list>
#include <solvers/smt/smt_conv.h>
#include <string>
#include <unistd.h>
#include <util/irep2.h>

/** Identifiers for SMT functions.
 *  Each SMT function gets a unique identifier, representing its interpretation
 *  when applied to some arguments. This can be used to describe a function
 *  application when joined with some arguments. Initial values such as
 *  terminal functions (i.e. bool, int, symbol literals) shouldn't normally
 *  be encountered and instead converted to an smt_ast before use. The
 *  'HACKS' function represents some kind of special case, according to where
 *  it is encountered; the same for 'INVALID'.
 *
 *  @see smt_convt::convert_terminal
 *  @see smt_convt::convert_ast
 */
enum smt_func_kind
{
  // Terminals
  SMT_FUNC_INT,
  SMT_FUNC_BOOL,
  SMT_FUNC_BVINT,
  SMT_FUNC_REAL,
  SMT_FUNC_SYMBOL,

  // Nonterminals
  SMT_FUNC_ADD,
  SMT_FUNC_BVADD,
  SMT_FUNC_SUB,
  SMT_FUNC_BVSUB,
  SMT_FUNC_MUL,
  SMT_FUNC_BVMUL,
  SMT_FUNC_DIV,
  SMT_FUNC_BVUDIV,
  SMT_FUNC_BVSDIV,
  SMT_FUNC_MOD,
  SMT_FUNC_BVSMOD,
  SMT_FUNC_BVUMOD,
  SMT_FUNC_SHL,
  SMT_FUNC_BVSHL,
  SMT_FUNC_BVASHR,
  SMT_FUNC_NEG,
  SMT_FUNC_BVNEG,
  SMT_FUNC_BVLSHR,
  SMT_FUNC_BVNOT,
  SMT_FUNC_BVNXOR,
  SMT_FUNC_BVNOR,
  SMT_FUNC_BVNAND,
  SMT_FUNC_BVXOR,
  SMT_FUNC_BVOR,
  SMT_FUNC_BVAND,

  // Logic
  SMT_FUNC_IMPLIES,
  SMT_FUNC_XOR,
  SMT_FUNC_OR,
  SMT_FUNC_AND,
  SMT_FUNC_NOT,

  // Comparisons
  SMT_FUNC_LT,
  SMT_FUNC_BVSLT,
  SMT_FUNC_BVULT,
  SMT_FUNC_GT,
  SMT_FUNC_BVSGT,
  SMT_FUNC_BVUGT,
  SMT_FUNC_LTE,
  SMT_FUNC_BVSLTE,
  SMT_FUNC_BVULTE,
  SMT_FUNC_GTE,
  SMT_FUNC_BVSGTE,
  SMT_FUNC_BVUGTE,

  SMT_FUNC_EQ,
  SMT_FUNC_NOTEQ,

  SMT_FUNC_ITE,

  SMT_FUNC_STORE,
  SMT_FUNC_SELECT,

  SMT_FUNC_CONCAT,
  SMT_FUNC_EXTRACT, // Not for going through mk app due to sillyness.

  SMT_FUNC_INT2REAL,
  SMT_FUNC_REAL2INT,
  SMT_FUNC_IS_INT,

  // floatbv operations
  SMT_FUNC_FNEG,
  SMT_FUNC_FABS,
  SMT_FUNC_ISZERO,
  SMT_FUNC_ISNAN,
  SMT_FUNC_ISINF,
  SMT_FUNC_ISNORMAL,
  SMT_FUNC_ISNEG,
  SMT_FUNC_ISPOS,
  SMT_FUNC_IEEE_EQ,
  SMT_FUNC_IEEE_ADD,
  SMT_FUNC_IEEE_SUB,
  SMT_FUNC_IEEE_MUL,
  SMT_FUNC_IEEE_DIV,
  SMT_FUNC_IEEE_FMA,
  SMT_FUNC_IEEE_SQRT,

  SMT_FUNC_IEEE_RM_NE,
  SMT_FUNC_IEEE_RM_ZR,
  SMT_FUNC_IEEE_RM_PI,
  SMT_FUNC_IEEE_RM_MI,

  SMT_FUNC_BV2FLOAT,
  SMT_FUNC_FLOAT2BV,
};

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
  std::vector<smt_astt> args;
};

class smtlib_convt : public smt_convt, public array_iface, public fp_convt
{
public:
  smtlib_convt(bool int_encoding, const namespacet &_ns);
  ~smtlib_convt() override;

  resultt dec_solve() override;
  const std::string solver_text() override;

  smt_astt mk_add(smt_astt a, smt_astt b) override;
  smt_astt mk_bvadd(smt_astt a, smt_astt b) override;
  smt_astt mk_sub(smt_astt a, smt_astt b) override;
  smt_astt mk_bvsub(smt_astt a, smt_astt b) override;
  smt_astt mk_mul(smt_astt a, smt_astt b) override;
  smt_astt mk_bvmul(smt_astt a, smt_astt b) override;
  smt_astt mk_mod(smt_astt a, smt_astt b) override;
  smt_astt mk_bvsmod(smt_astt a, smt_astt b) override;
  smt_astt mk_bvumod(smt_astt a, smt_astt b) override;
  smt_astt mk_div(smt_astt a, smt_astt b) override;
  smt_astt mk_bvsdiv(smt_astt a, smt_astt b) override;
  smt_astt mk_bvudiv(smt_astt a, smt_astt b) override;
  smt_astt mk_shl(smt_astt a, smt_astt b) override;
  smt_astt mk_bvshl(smt_astt a, smt_astt b) override;
  smt_astt mk_bvashr(smt_astt a, smt_astt b) override;
  smt_astt mk_bvlshr(smt_astt a, smt_astt b) override;
  smt_astt mk_neg(smt_astt a) override;
  smt_astt mk_bvneg(smt_astt a) override;
  smt_astt mk_bvnot(smt_astt a) override;
  smt_astt mk_bvnxor(smt_astt a, smt_astt b) override;
  smt_astt mk_bvnor(smt_astt a, smt_astt b) override;
  smt_astt mk_bvnand(smt_astt a, smt_astt b) override;
  smt_astt mk_bvxor(smt_astt a, smt_astt b) override;
  smt_astt mk_bvor(smt_astt a, smt_astt b) override;
  smt_astt mk_bvand(smt_astt a, smt_astt b) override;
  smt_astt mk_implies(smt_astt a, smt_astt b) override;
  smt_astt mk_xor(smt_astt a, smt_astt b) override;
  smt_astt mk_or(smt_astt a, smt_astt b) override;
  smt_astt mk_and(smt_astt a, smt_astt b) override;
  smt_astt mk_not(smt_astt a) override;
  smt_astt mk_lt(smt_astt a, smt_astt b) override;
  smt_astt mk_bvult(smt_astt a, smt_astt b) override;
  smt_astt mk_bvslt(smt_astt a, smt_astt b) override;
  smt_astt mk_gt(smt_astt a, smt_astt b) override;
  smt_astt mk_bvugt(smt_astt a, smt_astt b) override;
  smt_astt mk_bvsgt(smt_astt a, smt_astt b) override;
  smt_astt mk_le(smt_astt a, smt_astt b) override;
  smt_astt mk_bvule(smt_astt a, smt_astt b) override;
  smt_astt mk_bvsle(smt_astt a, smt_astt b) override;
  smt_astt mk_ge(smt_astt a, smt_astt b) override;
  smt_astt mk_bvuge(smt_astt a, smt_astt b) override;
  smt_astt mk_bvsge(smt_astt a, smt_astt b) override;
  smt_astt mk_eq(smt_astt a, smt_astt b) override;
  smt_astt mk_store(smt_astt a, smt_astt b, smt_astt c) override;
  smt_astt mk_select(smt_astt a, smt_astt b) override;
  smt_astt mk_real2int(smt_astt a) override;
  smt_astt mk_int2real(smt_astt a) override;
  smt_astt mk_isint(smt_astt a) override;

  void assert_ast(const smt_ast *a) override;

  smt_sortt mk_bool_sort() override;
  smt_sortt mk_real_sort() override;
  smt_sortt mk_int_sort() override;
  smt_sortt mk_bv_sort(std::size_t width) override;
  smt_sortt mk_fbv_sort(std::size_t width) override;
  smt_sortt mk_array_sort(smt_sortt domain, smt_sortt range) override;
  smt_sortt mk_bvfp_sort(std::size_t ew, std::size_t sw) override;
  smt_sortt mk_bvfp_rm_sort() override;

  smt_ast *mk_smt_int(const mp_integer &theint, bool sign) override;
  smt_ast *mk_smt_real(const std::string &str) override;
  smt_astt mk_smt_bv(const mp_integer &theint, smt_sortt s) override;
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
  smt_astt mk_concat(smt_astt a, smt_astt b) override;
  smt_astt mk_ite(smt_astt cond, smt_astt t, smt_astt f) override;

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

  /** Mapping of SMT function IDs to their names. XXX, incorrect size. */
  static const std::string smt_func_name_table[expr2t::end_expr_id];
};

#endif /* _ESBMC_SOLVERS_SMTLIB_SMTLIB_CONV_H */
