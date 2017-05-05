#ifdef WITH_PYTHON
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_tuple.h>
#include <solvers/smt/smt_array.h>
#include <boost/python/class.hpp>

class smt_sort_wrapper : public smt_sort, public boost::python::wrapper<smt_sort>
{
public:
  friend class get_override_checked_class;
  template <typename ...Args>
  smt_sort_wrapper(Args ...args);

  static boost::python::object cast_sort_down(smt_sortt s);

  ~smt_sort_wrapper() {}
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
  smt_astt update(smt_convt *ctx, smt_astt value, unsigned int idx, expr2tc idx_expr = expr2tc()) const;
  smt_astt default_update(smt_convt *ctx, smt_astt value, unsigned int idx, expr2tc idx_expr = expr2tc()) const;
  smt_astt select(smt_convt *ctx, const expr2tc &idx) const;
  smt_astt default_select(smt_convt *ctx, const expr2tc &idx) const;
  smt_astt project(smt_convt *ctx, unsigned int elem) const;
  smt_astt default_project(smt_convt *ctx, unsigned int elem) const;

  void dump() const {
    std::cerr << "smt_ast_wrapper::dump: unimplemented" << std::endl;
    abort();
  }
};

class smt_convt_wrapper : public smt_convt, public array_iface, public tuple_iface, public boost::python::wrapper<smt_convt>
{
public:
  friend class get_override_checked_class;
  friend class smt_convt_wrapper_cvt;
  smt_convt_wrapper(bool int_encoding, const namespacet &_ns, bool bools_in_arrays, bool can_init_inf_arrays);
  static boost::python::object cast_conv_down(smt_convt *c);
  smt_astt mk_func_app(smt_sortt s, smt_func_kind k, smt_astt const *args, unsigned int numargs);
  smt_astt mk_func_app_remangled(smt_sortt s, smt_func_kind k, boost::python::object o);
  void assert_ast(smt_astt a);
  smt_convt::resultt dec_solve();
  const std::string solver_text();
  tvt l_get(smt_astt a);
  smt_sortt mk_sort(const smt_sort_kind k, ...);
  smt_sortt mk_sort_remangled(boost::python::object o);
  smt_astt mk_smt_int(const mp_integer &theint, bool sign);
  smt_astt mk_smt_bool(bool val);
  smt_astt mk_smt_symbol(const std::string &name, smt_sortt s);
  smt_astt mk_smt_real(const std::string &str);
  smt_astt mk_smt_bvint(const mp_integer &theint, bool sign, unsigned int w);
  expr2tc get_bool(smt_astt a);
  expr2tc get_bv(const type2tc &t, smt_astt a);
  smt_astt mk_extract(smt_astt a, unsigned int high, unsigned int low, smt_sortt s);
  /*************************** Array API ***********************************/
  smt_astt mk_array_symbol(const std::string &name, smt_sortt sort, smt_sortt subtype);
  expr2tc get_array_elem(smt_astt a, uint64_t idx, const type2tc &subtype);
  const smt_ast * convert_array_of(smt_astt init_val, unsigned long domain_width);
  void add_array_constraints_for_solving();
  void push_array_ctx(void);
  void pop_array_ctx(void);
  /*************************** Tuple API ***********************************/
  smt_sortt mk_struct_sort(const type2tc &type);
  smt_astt tuple_create(const expr2tc &structdef);
  smt_astt tuple_fresh(smt_sortt s, std::string name = "");
  smt_astt tuple_array_create(const type2tc &array_type, smt_astt *inputargs,
      bool const_array, smt_sortt domain);
  smt_astt tuple_array_create_remangled(const type2tc &array_type,
      boost::python::object l, bool const_array, smt_sortt domain);
  smt_astt tuple_array_of(const expr2tc &init_value, unsigned long domain_width);
  smt_astt mk_tuple_symbol(const std::string &name, smt_sortt s);
  smt_astt mk_tuple_array_symbol(const expr2tc &expr);
  expr2tc tuple_get(const expr2tc &expr);
  void add_tuple_constraints_for_solving();
  void push_tuple_ctx();
  void pop_tuple_ctx();

  // Uhhhh, float api?
  smt_ast *mk_smt_bvfloat(const ieee_floatt &thereal, unsigned ew, unsigned sw);
  smt_astt mk_smt_bvfloat_nan(unsigned ew, unsigned sw);
  smt_astt mk_smt_bvfloat_inf(bool sgn, unsigned ew, unsigned sw);
  smt_astt mk_smt_bvfloat_rm(ieee_floatt::rounding_modet rm);
  smt_astt mk_smt_typecast_from_bvfloat(const typecast2t &cast);
  smt_astt mk_smt_typecast_to_bvfloat(const typecast2t &cast);
  smt_astt mk_smt_nearbyint_from_float(const nearbyint2t &expr);
  smt_astt mk_smt_bvfloat_arith_ops(const expr2tc &expr);
};

#endif /* WITH_PYTHON */
