#ifdef WITH_PYTHON
#include <solvers/smt/smt_conv.h>
#include <boost/python/class.hpp>

class smt_sort_wrapper : public smt_sort,
                         public boost::python::wrapper<smt_sort>
{
public:
  friend class get_override_checked_class;
  template <typename... Args>
  smt_sort_wrapper(Args... args);

  static boost::python::object cast_sort_down(smt_sortt s);

  ~smt_sort_wrapper()
  {
  }
};

class smt_ast_wrapper : public smt_ast, public boost::python::wrapper<smt_ast>
{
public:
  friend class get_override_checked_class;
  smt_ast_wrapper(smt_convt *ctx, smt_sortt s);

  static boost::python::object cast_ast_down(smt_astt a);

  smt_astt ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const;
  smt_astt default_ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const;
  smt_astt eq(smt_convt *ctx, smt_astt other) const;
  smt_astt default_eq(smt_convt *ctx, smt_astt other) const;
  void assign(smt_convt *ctx, smt_astt sym) const;
  void default_assign(smt_convt *ctx, smt_astt sym) const;
  smt_astt update(
    smt_convt *ctx,
    smt_astt value,
    unsigned int idx,
    expr2tc idx_expr = expr2tc()) const;
  smt_astt default_update(
    smt_convt *ctx,
    smt_astt value,
    unsigned int idx,
    expr2tc idx_expr = expr2tc()) const;
  smt_astt select(smt_convt *ctx, const expr2tc &idx) const;
  smt_astt default_select(smt_convt *ctx, const expr2tc &idx) const;
  smt_astt project(smt_convt *ctx, unsigned int elem) const;
  smt_astt default_project(smt_convt *ctx, unsigned int elem) const;

  void dump() const
  {
    std::cerr << "smt_ast_wrapper::dump: unimplemented" << std::endl;
    abort();
  }
};

class smt_convt_wrapper : public smt_convt,
                          public array_iface,
                          public tuple_iface,
                          public boost::python::wrapper<smt_convt>
{
public:
  friend class get_override_checked_class;
  friend class smt_convt_wrapper_cvt;
  smt_convt_wrapper(
    bool int_encoding,
    const namespacet &_ns,
    bool bools_in_arrays,
    bool can_init_inf_arrays);
  static boost::python::object cast_conv_down(smt_convt *c);
  void assert_ast(smt_astt a);
  smt_convt::resultt dec_solve();
  const std::string solver_text();
  tvt l_get(smt_astt a);

  smt_astt mk_smt_int(const mp_integer &theint, bool sign);
  smt_astt mk_smt_bool(bool val);
  smt_astt mk_smt_symbol(const std::string &name, smt_sortt s);
  smt_astt mk_smt_real(const std::string &str);
  smt_astt mk_smt_bv(const mp_integer &theint, smt_sortt s);

  smt_astt mk_add(smt_astt a, smt_astt b);
  smt_astt mk_bvadd(smt_astt a, smt_astt b);
  smt_astt mk_sub(smt_astt a, smt_astt b);
  smt_astt mk_bvsub(smt_astt a, smt_astt b);
  smt_astt mk_mul(smt_astt a, smt_astt b);
  smt_astt mk_bvmul(smt_astt a, smt_astt b);
  smt_astt mk_mod(smt_astt a, smt_astt b);
  smt_astt mk_bvsmod(smt_astt a, smt_astt b);
  smt_astt mk_bvumod(smt_astt a, smt_astt b);
  smt_astt mk_div(smt_astt a, smt_astt b);
  smt_astt mk_bvsdiv(smt_astt a, smt_astt b);
  smt_astt mk_bvudiv(smt_astt a, smt_astt b);
  smt_astt mk_shl(smt_astt a, smt_astt b);
  smt_astt mk_bvshl(smt_astt a, smt_astt b);
  smt_astt mk_bvashr(smt_astt a, smt_astt b);
  smt_astt mk_bvlshr(smt_astt a, smt_astt b);
  smt_astt mk_neg(smt_astt a);
  smt_astt mk_bvneg(smt_astt a);
  smt_astt mk_bvnot(smt_astt a);
  smt_astt mk_bvnxor(smt_astt a, smt_astt b);
  smt_astt mk_bvnor(smt_astt a, smt_astt b);
  smt_astt mk_bvnand(smt_astt a, smt_astt b);
  smt_astt mk_bvxor(smt_astt a, smt_astt b);
  smt_astt mk_bvor(smt_astt a, smt_astt b);
  smt_astt mk_bvand(smt_astt a, smt_astt b);
  smt_astt mk_implies(smt_astt a, smt_astt b);
  smt_astt mk_xor(smt_astt a, smt_astt b);
  smt_astt mk_or(smt_astt a, smt_astt b);
  smt_astt mk_and(smt_astt a, smt_astt b);
  smt_astt mk_not(smt_astt a);
  smt_astt mk_lt(smt_astt a, smt_astt b);
  smt_astt mk_bvult(smt_astt a, smt_astt b);
  smt_astt mk_bvslt(smt_astt a, smt_astt b);
  smt_astt mk_gt(smt_astt a, smt_astt b);
  smt_astt mk_bvugt(smt_astt a, smt_astt b);
  smt_astt mk_bvsgt(smt_astt a, smt_astt b);
  smt_astt mk_le(smt_astt a, smt_astt b);
  smt_astt mk_bvule(smt_astt a, smt_astt b);
  smt_astt mk_bvsle(smt_astt a, smt_astt b);
  smt_astt mk_ge(smt_astt a, smt_astt b);
  smt_astt mk_bvuge(smt_astt a, smt_astt b);
  smt_astt mk_bvsge(smt_astt a, smt_astt b);
  smt_astt mk_eq(smt_astt a, smt_astt b);
  smt_astt mk_neq(smt_astt a, smt_astt b);
  smt_astt mk_store(smt_astt a, smt_astt b, smt_astt c);
  smt_astt mk_select(smt_astt a, smt_astt b);
  smt_astt mk_real2int(smt_astt a);
  smt_astt mk_int2real(smt_astt a);
  smt_astt mk_isint(smt_astt a);
  smt_astt mk_extract(smt_astt a, unsigned int high, unsigned int low);
  smt_astt mk_concat(smt_astt a, smt_astt b);
  smt_astt mk_ite(smt_astt cond, smt_astt t, smt_astt f);
  smt_astt mk_sign_ext(smt_astt a, unsigned int topwidth);
  smt_astt mk_zero_ext(smt_astt a, unsigned int topwidth);

  bool get_bool(smt_astt a);
  BigInt get_bv(smt_astt a);

  smt_sortt mk_bool_sort();
  smt_sortt mk_real_sort();
  smt_sortt mk_int_sort();
  smt_sortt mk_bv_sort(std::size_t width);
  smt_sortt mk_fbv_sort(std::size_t width);
  smt_sortt mk_bvfp_sort(std::size_t ew, std::size_t sw);
  smt_sortt mk_bvfp_rm_sort();
  smt_sortt mk_array_sort(smt_sortt domain, smt_sortt range);

  /*************************** Array API ***********************************/
  smt_astt
  mk_array_symbol(const std::string &name, smt_sortt sort, smt_sortt subtype);
  expr2tc get_array_elem(smt_astt a, uint64_t idx, const type2tc &subtype);
  const smt_ast *
  convert_array_of(smt_astt init_val, unsigned long domain_width);
  void add_array_constraints_for_solving();
  void push_array_ctx(void);
  void pop_array_ctx(void);
  /*************************** Tuple API ***********************************/
  smt_sortt mk_struct_sort(const type2tc &type);
  smt_astt tuple_create(const expr2tc &structdef);
  smt_astt tuple_fresh(smt_sortt s, std::string name = "");
  smt_astt tuple_array_create(
    const type2tc &array_type,
    smt_astt *inputargs,
    bool const_array,
    smt_sortt domain);
  smt_astt tuple_array_create_remangled(
    const type2tc &array_type,
    boost::python::object l,
    bool const_array,
    smt_sortt domain);
  smt_astt
  tuple_array_of(const expr2tc &init_value, unsigned long domain_width);
  smt_astt mk_tuple_symbol(const std::string &name, smt_sortt s);
  smt_astt mk_tuple_array_symbol(const expr2tc &expr);
  expr2tc tuple_get(const expr2tc &expr);
  void add_tuple_constraints_for_solving();
  void push_tuple_ctx();
  void pop_tuple_ctx();

  smt_astt mk_smt_fpbv(const ieee_floatt &thereal);
  smt_sortt mk_fpbv_sort(const unsigned ew, const unsigned sw);
  smt_sortt mk_fpbv_rm_sort();
  smt_astt mk_smt_fpbv_nan(unsigned ew, unsigned sw);
  smt_astt mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw);
  smt_astt mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm);
  smt_astt mk_smt_typecast_from_fpbv_to_ubv(smt_astt from, std::size_t width);
  smt_astt mk_smt_typecast_from_fpbv_to_sbv(smt_astt from, std::size_t width);
  smt_astt
  mk_smt_typecast_from_fpbv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm);
  smt_astt
  mk_smt_typecast_ubv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm);
  smt_astt
  mk_smt_typecast_sbv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm);
  smt_astt mk_smt_nearbyint_from_float(smt_astt from, smt_astt rm);
  smt_astt mk_smt_fpbv_add(smt_astt lhs, smt_astt rhs, smt_astt rm);
  smt_astt mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm);
  smt_astt mk_smt_fpbv_mul(smt_astt lhs, smt_astt rhs, smt_astt rm);
  smt_astt mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm);
  smt_astt mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm);
  smt_astt mk_smt_fpbv_fma(smt_astt v1, smt_astt v2, smt_astt v3, smt_astt rm);
  smt_astt mk_smt_fpbv_eq(smt_astt lhs, smt_astt rhs);
  smt_astt mk_smt_fpbv_gt(smt_astt lhs, smt_astt rhs);
  smt_astt mk_smt_fpbv_lt(smt_astt lhs, smt_astt rhs);
  smt_astt mk_smt_fpbv_gte(smt_astt lhs, smt_astt rhs);
  smt_astt mk_smt_fpbv_lte(smt_astt lhs, smt_astt rhs);
  smt_astt mk_smt_fpbv_is_nan(smt_astt op);
  smt_astt mk_smt_fpbv_is_inf(smt_astt op);
  smt_astt mk_smt_fpbv_is_normal(smt_astt op);
  smt_astt mk_smt_fpbv_is_zero(smt_astt op);
  smt_astt mk_smt_fpbv_is_negative(smt_astt op);
  smt_astt mk_smt_fpbv_is_positive(smt_astt op);
  smt_astt mk_smt_fpbv_abs(smt_astt op);
  smt_astt mk_smt_fpbv_neg(smt_astt op);
  ieee_floatt get_fpbv(smt_astt a);
  smt_astt mk_from_bv_to_fp(smt_astt op, smt_sortt to);
  smt_astt mk_from_fp_to_bv(smt_astt op);
};

#endif /* WITH_PYTHON */
