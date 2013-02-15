#ifndef _ESBMC_PROP_SMT_SMT_CONV_H_
#define _ESBMC_PROP_SMT_SMT_CONV_H_

#include <irep2.h>
#include <namespace.h>

#include <util/pointer_offset_size.h>

#include <solvers/prop/prop_conv.h>
#include <solvers/prop/pointer_logic.h>

enum smt_sort_kind {
  SMT_SORT_INT = 1,
  SMT_SORT_REAL = 2,
  SMT_SORT_BV = 4,
  SMT_SORT_ARRAY = 8,
  SMT_SORT_BOOL = 16,
  SMT_SORT_STRUCT = 32,
  SMT_SORT_UNION = 64, // Contencious
};

#define SMT_SORT_ALLINTS (SMT_SORT_INT | SMT_SORT_REAL | SMT_SORT_BV)

enum smt_func_kind {
  // Terminals
  SMT_FUNC_HACKS = 0, // indicate the solver /has/ to use the temp expr.
  SMT_FUNC_INVALID = 1, // For conversion lookup table only
  SMT_FUNC_INT = 2,
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
  SMT_FUNC_BVDIV,
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

  SMT_FUNC_INT2REAL,
  SMT_FUNC_REAL2INT,
};

class smt_sort {
  // Same story as smt_ast.
public:
  smt_sort_kind id;
  smt_sort(smt_sort_kind i) : id(i) { }
};

class smt_ast {
  // Mostly opaque class for storing whatever data the backend solver feels
  // is appropriate. I orignally thought this should contain useful tracking
  // data for what kind of AST this is, but then realised that this is just
  // overhead and whatever the solver stores should be enough for that too
  //
  // Question - should this AST node (or what it represents) not be the same as
  // any other expression in ESBMC and just use the existing expr
  // representations?  Answer - Perhaps, but there's a semantic shift between
  // what ESBMC uses expressions for and what the SMT language actually /has/,
  // i.e. just some function applications and suchlike. Plus an additional
  // layer of type safety can be applied here that can't in the expression
  // stuff.
  //
  // In particular, things where multiple SMT functions map onto one expression
  // operator are troublesome. This means multiple integer operators for
  // different integer modes, signed and unsigned comparisons, the multitude of
  // things that an address-of can turn into, and so forth.

  // We /do/ need some sort information in conversion.
public:
  const smt_sort *sort;

  smt_ast(const smt_sort *s) : sort(s) { }

};

class smt_convt: public prop_convt
{
public:
  smt_convt(bool enable_cache, bool int_encoding, const namespacet &_ns,
            bool is_cpp);
  ~smt_convt();

  virtual void push_ctx(void);
  virtual void pop_ctx(void);

  virtual void assert_lit(const literalt &l) = 0;

  virtual smt_ast *mk_func_app(const smt_sort *s, smt_func_kind k,
                               const smt_ast **args, unsigned int numargs,
                               const expr2tc &temp) = 0;
  virtual smt_sort *mk_sort(const smt_sort_kind k, ...) = 0;
  virtual literalt mk_lit(const smt_ast *s) = 0;
  virtual smt_ast *mk_smt_int(const mp_integer &theint, bool sign, const expr2tc &t) = 0;
  virtual smt_ast *mk_smt_real(const mp_integer &thereal, const expr2tc &t) = 0;
  virtual smt_ast *mk_smt_bvint(const mp_integer &theint, bool sign, unsigned int w, const expr2tc &t) = 0;
  virtual smt_ast *mk_smt_bool(bool val, const expr2tc &t) = 0;
  virtual smt_ast *mk_smt_symbol(const std::string &name, const smt_sort *s, const expr2tc &t) =0;
  virtual smt_sort *mk_struct_sort(const type2tc &type) = 0;
  // XXX XXX XXX -- turn this into a formulation on top of structs.
  virtual smt_sort *mk_union_sort(const type2tc &type) = 0;
  virtual smt_ast *mk_extract(const smt_ast *a, unsigned int high,
                              unsigned int low, const smt_sort *s,
                              const expr2tc &tmp) = 0;

  virtual void set_to(const expr2tc &expr, bool value);
  virtual literalt convert_expr(const expr2tc &expr);

  // Things that the SMT converter can flatten to SMT, but that the specific
  // solver being used might have its own support for (in which case it should
  // override the below).
  virtual smt_ast *tuple_create(const expr2tc &structdef);
  virtual smt_ast *tuple_project(const smt_ast *a, const smt_sort *s, unsigned int field, const expr2tc &tmp);
  virtual smt_ast *tuple_update(const smt_ast *a, unsigned int field,
                                const smt_ast *val, const expr2tc &tmp);
  virtual smt_ast *tuple_equality(const smt_ast *a, const smt_ast *b, const expr2tc &tmp);
  virtual smt_ast *tuple_ite(const smt_ast *cond, const smt_ast *trueval,
                             const smt_ast *false_val, const smt_sort *sort,
                             const expr2tc &expr);
  virtual smt_ast *overflow_arith(const expr2tc &expr);
  virtual smt_ast *overflow_cast(const expr2tc &expr);
  virtual smt_ast *overflow_neg(const expr2tc &expr);

  // Internal foo

  smt_sort *convert_sort(const type2tc &type);
  smt_ast *convert_terminal(const expr2tc &expr);
  const smt_ast *convert_ast(const expr2tc &expr);
  const smt_ast *convert_pointer_arith(const expr2tc &expr, const type2tc &t);
  const smt_ast *convert_addr_of(const expr2tc &expr);
  const smt_ast *convert_identifier_pointer(const expr2tc &expr,
                                            std::string sym);
  const smt_ast *convert_sign_ext(const smt_ast *a, const smt_sort *s,
                                  unsigned int topbit, unsigned int topwidth);
  const smt_ast *convert_zero_ext(const smt_ast *a, const smt_sort *s,
                                  unsigned int topwidth);
  const smt_ast *convert_is_nan(const expr2tc &expr, const smt_ast *oper);
  const smt_ast *convert_byte_extract(const expr2tc &expr);
  const smt_ast *convert_byte_update(const expr2tc &expr);
  void assert_expr(const expr2tc &e);

  // To be called by the subclass when all other construction has occured.
  // It needs to be able to convert a variety of things, which need to be
  // available.
  void init_addr_space_array(void);
  void bump_addrspace_array(unsigned int idx, const expr2tc &val);
  std::string get_cur_addrspace_ident(void);
  void finalize_pointer_chain(unsigned int obj_num);

  const smt_ast *convert_typecast_bool(const typecast2t &cast);
  const smt_ast *convert_typecast_fixedbv_nonint(const expr2tc &cast);
  const smt_ast *convert_typecast_to_ints(const typecast2t &cast);
  const smt_ast *convert_typecast_to_ptr(const typecast2t &cast);
  const smt_ast *convert_typecast_from_ptr(const typecast2t &cast);
  const smt_ast *convert_typecast_struct(const typecast2t &cast);
  const smt_ast *convert_typecast(const expr2tc &expr);

  // Types

  // Types for union map.
  struct union_var_mapt {
    std::string ident;
    unsigned int idx;
    unsigned int level;
  };

  typedef boost::multi_index_container<
    union_var_mapt,
    boost::multi_index::indexed_by<
      boost::multi_index::hashed_unique<
        BOOST_MULTI_INDEX_MEMBER(union_var_mapt, std::string, ident)
      >,
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(union_var_mapt, unsigned int, level),
        std::greater<unsigned int>
      >
    >
  > union_varst;

  // Type for (optional) AST cache

  struct smt_cache_entryt {
    const expr2tc val;
    const smt_ast *ast;
    unsigned int level;
  };

  typedef boost::multi_index_container<
    smt_cache_entryt,
    boost::multi_index::indexed_by<
      boost::multi_index::hashed_unique<
        BOOST_MULTI_INDEX_MEMBER(smt_cache_entryt, const expr2tc, val)
      >,
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(smt_cache_entryt, unsigned int, level),
        std::greater<unsigned int>
      >
    >
  > smt_cachet;

  struct expr_op_convert {
    smt_func_kind int_mode_func;
    smt_func_kind bv_mode_func_signed;
    smt_func_kind bv_mode_func_unsigned;
    unsigned int args;
    unsigned long permitted_sorts;
  };

  // Members
  union_varst union_vars;
  smt_cachet smt_cache;
  std::list<pointer_logict> pointer_logic;
  type2tc pointer_struct;
  const struct_type2t *pointer_type_data; // ptr of pointer_struct
  bool caching;
  bool int_encoding;
  const namespacet &ns;
  std::string dyn_info_arr_name;

  std::list<unsigned int> addr_space_sym_num;
  type2tc addr_space_type;
  const struct_type2t *addr_space_type_data;
  type2tc addr_space_arr_type;
  std::list<std::map<unsigned, unsigned> > addr_space_data;

  static const expr_op_convert smt_convert_table[expr2t::end_expr_id];
};

#endif /* _ESBMC_PROP_SMT_SMT_CONV_H_ */
