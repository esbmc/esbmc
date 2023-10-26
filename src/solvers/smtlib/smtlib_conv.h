#ifndef _ESBMC_SOLVERS_SMTLIB_SMTLIB_CONV_H
#define _ESBMC_SOLVERS_SMTLIB_SMTLIB_CONV_H

#include <list>
#include <solvers/smt/smt_conv.h>
#include <string>
#ifndef _WIN32
#  include <unistd.h>
#endif
#include <irep2/irep2.h>

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
  SMT_FUNC_FORALL,
  SMT_FUNC_EXISTS,
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

struct sexpr
{
  sexpr() = default;
  sexpr(sexpr &&) noexcept = default;
  sexpr &operator=(sexpr &&) noexcept = default;

  int token = 0; // If zero, then an sexpr list
  std::list<sexpr> sexpr_list;
  std::string data; // Text rep of parsed token.
};

static_assert(std::is_nothrow_move_constructible_v<sexpr>);
static_assert(std::is_nothrow_move_assignable_v<sexpr>);

class smtlib_smt_sort : public smt_sort
{
public:
  explicit smtlib_smt_sort(smt_sort_kind k, unsigned int w) : smt_sort(k, w){};
  explicit smtlib_smt_sort(smt_sort_kind k, unsigned int w1, unsigned int w2)
    : smt_sort(k, w1, w2){};
  explicit smtlib_smt_sort(smt_sort_kind k) : smt_sort(k)
  {
  }
  explicit smtlib_smt_sort(
    smt_sort_kind k,
    const smtlib_smt_sort *dom,
    const smtlib_smt_sort *rag)
    : smt_sort(k, dom->get_data_width(), rag), domain(dom), range(rag)
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

  void dump() const override;

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
  smtlib_convt(const namespacet &_ns, const optionst &options);
  ~smtlib_convt() override;

  resultt dec_solve() override;
  const std::string solver_text() override;

  void clear();
  void output_content(std::string& cont);
  std::string get_file_contents();

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
  smt_astt mk_forall(smt_astt a, smt_astt b) override;
  smt_astt mk_exists(smt_astt a, smt_astt b) override;
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

  void assert_ast(smt_astt a) override;

  smt_sortt mk_bool_sort() override;
  smt_sortt mk_real_sort() override;
  smt_sortt mk_int_sort() override;
  smt_sortt mk_bv_sort(std::size_t width) override;
  smt_sortt mk_fbv_sort(std::size_t width) override;
  smt_sortt mk_array_sort(smt_sortt domain, smt_sortt range) override;
  smt_sortt mk_bvfp_sort(std::size_t ew, std::size_t sw) override;
  smt_sortt mk_bvfp_rm_sort() override;

  smt_astt mk_smt_int(const BigInt &theint) override;
  smt_astt mk_smt_real(const std::string &str) override;
  smt_astt mk_smt_bv(const BigInt &theint, smt_sortt s) override;
  smt_astt mk_smt_bool(bool val) override;
  smt_astt mk_smt_symbol(const std::string &name, const smt_sort *s) override;
  smt_astt mk_array_symbol(
    const std::string &name,
    const smt_sort *s,
    smt_sortt array_subtype) override;
  smt_sort *mk_struct_sort(const type2tc &type);
  smt_astt mk_extract(smt_astt a, unsigned int high, unsigned int low) override;
  smt_astt mk_sign_ext(smt_astt a, unsigned int topwidth) override;
  smt_astt mk_zero_ext(smt_astt a, unsigned int topwidth) override;
  smt_astt mk_concat(smt_astt a, smt_astt b) override;
  smt_astt mk_ite(smt_astt cond, smt_astt t, smt_astt f) override;

  smt_astt
  convert_array_of(smt_astt init_val, unsigned long domain_width) override;

  sexpr get_value(smt_astt a) const;

  bool get_bool(smt_astt a) override;
  tvt l_get(smt_astt a) override;
  BigInt get_bv(smt_astt a, bool is_signed) override;
  expr2tc
  get_array_elem(smt_astt array, uint64_t index, const type2tc &type) override;

  std::string sort_to_string(const smt_sort *s) const;

  unsigned int
  emit_terminal_ast(const smtlib_smt_ast *a, std::string &output) const;

  unsigned int emit_ast(
    const smtlib_smt_ast *ast,
    std::string &output,
    std::unordered_map<const smtlib_smt_ast *, std::string> &temp_symbols)
    const;

  void emit_ast(const smtlib_smt_ast *ast) const;

  void push_ctx() override;
  void pop_ctx() override;

  void dump_smt() override;

  template <typename... Ts>
  void emit(const Ts &...) const;
  void flush() const;

  // Members

  struct process_emitter
  {
    FILE *out_stream;
    FILE *in_stream;
    void *org_sigpipe_handler; /* TODO: static */

    std::string solver_name;
    std::string solver_version;

    explicit process_emitter(const std::string &cmd);
    process_emitter(const process_emitter &) = delete;

    ~process_emitter() noexcept;

    process_emitter &operator=(const process_emitter &) = delete;

    template <typename... Ts>
    void emit(const char *fmt, Ts &&...) const;
    void flush() const;

    explicit operator bool() const noexcept;
  } emit_proc;

  struct file_emitter
  {
    FILE *out_stream;
    std::string _path;

    explicit file_emitter(const std::string &path);
    file_emitter(const file_emitter &) = delete;

    ~file_emitter() noexcept;

    file_emitter &operator=(const file_emitter &) = delete;

    template <typename... Ts>
    void emit(const char *fmt, Ts &&...) const;
    void flush() const;

    void clear();
    void output_content(std::string cont);
    std::string get_file_contents();

    explicit operator bool() const noexcept;
  } emit_opt_output;

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

  typedef std::map<std::string, symbol_table_rec> map_tablet;
  typedef std::vector<map_tablet> vampire_tablet;

  bool vamp_for_loops;

  symbol_tablet symbol_table;
  vampire_tablet vampire_sym_table;

  static const std::string temp_prefix;

  struct external_process_died : std::runtime_error
  {
    using std::runtime_error::runtime_error;
  };
};

#endif /* _ESBMC_SOLVERS_SMTLIB_SMTLIB_CONV_H */
