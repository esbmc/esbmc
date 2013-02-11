#include "smt_conv.h"

smt_convt::smt_convt(bool enable_cache, bool intmode)
  : caching(enable_cache), int_encoding(intmode)
{
  std::vector<type2tc> members;
  std::vector<irep_idt> names;

  members.push_back(type_pool.get_uint(config.ansi_c.pointer_width));
  members.push_back(type_pool.get_uint(config.ansi_c.pointer_width));
  names.push_back(irep_idt("pointer_object"));
  names.push_back(irep_idt("pointer_offset"));

  pointer_struct = type2tc(new struct_type2t(members, names, "pointer_struct"));
}

smt_convt::~smt_convt(void)
{
}

void
smt_convt::push_ctx(void)
{
  prop_convt::push_ctx();
}

void
smt_convt::pop_ctx(void)
{
  prop_convt::pop_ctx();

  union_varst::nth_index<1>::type &union_numindex = union_vars.get<1>();
  union_numindex.erase(ctx_level);
  smt_cachet::nth_index<1>::type &cache_numindex = smt_cache.get<1>();
  cache_numindex.erase(ctx_level);
}

void
smt_convt::set_to(const expr2tc &expr, bool value)
{

  l_set_to(convert(expr), value);

  // Workaround for the fact that we don't have a good way of encoding unions
  // into SMT. Just work out what the last assigned field is.
  if (is_equality2t(expr) && value) {
    const equality2t eq = to_equality2t(expr);
    if (is_union_type(eq.side_1->type) && is_with2t(eq.side_2)) {
      const symbol2t sym = to_symbol2t(eq.side_1);
      const with2t with = to_with2t(eq.side_2);
      const union_type2t &type = to_union_type(eq.side_1->type);
      const std::string &ref = sym.get_symbol_name();
      const constant_string2t &str = to_constant_string2t(with.update_field);

      unsigned int idx = 0;
      forall_names(it, type.member_names) {
        if (*it == str.value)
          break;
        idx++;
      }

      assert(idx != type.member_names.size() &&
             "Member name of with expr not found in struct/union type");

      union_var_mapt mapentry = { ref, idx, 0 };
      union_vars.insert(mapentry);
    }
  }
}

literalt
smt_convt::convert_expr(const expr2tc &expr)
{
  const smt_ast *a = convert_ast(expr);
  return mk_lit(a);
}

const smt_ast *
smt_convt::convert_ast(const expr2tc &expr)
{
  const smt_ast *args[4];
  const smt_sort *sort;
  smt_ast *a;
  unsigned int num_args, used_sorts = 0;

  if (caching) {
    smt_cachet::const_iterator cache_result = smt_cache.find(expr);
    if (cache_result != smt_cache.end())
      return (cache_result->ast);
  }

  // Convert /all the arguments/.
  unsigned int i = 0;
  forall_operands2(it, idx, expr) {
    args[i] = convert_ast(*it);
    used_sorts |= args[i]->sort->id;
    i++;
  }
  num_args = i;

  sort = convert_sort(expr->type);

  const expr_op_convert *cvt = &smt_convert_table[expr->expr_id];

  if ((int_encoding && cvt->int_mode_func != SMT_FUNC_INVALID) ||
      (!int_encoding && cvt->bv_mode_func_signed != SMT_FUNC_INVALID)) {
    assert(cvt->args == num_args);
    // An obvious check, but catches cases where we add a field to a future expr
    // and then fail to update the SMT layer, leading to an ignored field.

    // Now check sort types.
    if ((used_sorts | cvt->permitted_sorts) == cvt->permitted_sorts) {
      // Matches; we can just convert this.
      smt_func_kind k = (int_encoding) ? cvt->int_mode_func
                      : (is_signedbv_type(expr->type))
                          ? cvt->bv_mode_func_signed
                          : cvt->bv_mode_func_unsigned;
      a = mk_func_app(sort, k, &args[0], cvt->args, expr);
      goto done;
    }
  }

  switch (expr->expr_id) {
  case expr2t::constant_int_id:
  case expr2t::constant_fixedbv_id:
  case expr2t::constant_bool_id:
  case expr2t::symbol_id:
    a = convert_terminal(expr);
    break;
  case expr2t::mul_id:
  {
    assert(!is_fixedbv_type(expr) && "haven't got SMT backend supporting fixedbv mul yet");
    assert(0);
  }
  case expr2t::div_id:
  {
    assert(!is_fixedbv_type(expr) && "haven't got SMT backend supporting fixedbv div yet");
    assert(0);
  }
  default:
    a = mk_func_app(sort, SMT_FUNC_HACKS, &args[0], 0, expr);
    break;
  }

done:
  struct smt_cache_entryt entry = { expr, a, ctx_level };
  smt_cache.insert(entry);
  return a;
}

smt_sort *
smt_convt::convert_sort(const type2tc &type)
{
  bool is_signed = true;

  switch (type->type_id) {
  case type2t::bool_id:
    return mk_sort(SMT_SORT_BOOL);
  case type2t::struct_id:
    return mk_struct_sort(type);
  case type2t::union_id:
    return mk_union_sort(type);
  case type2t::pointer_id:
    return mk_struct_sort(pointer_struct);
  case type2t::unsignedbv_id:
    is_signed = false;
    /* FALLTHROUGH */
  case type2t::signedbv_id:
  {
    unsigned int width = type->get_width();
    if (int_encoding)
      return mk_sort(SMT_SORT_INT, is_signed);
    else
      return mk_sort(SMT_SORT_BV, width, is_signed);
  }
  case type2t::fixedbv_id:
  {
    unsigned int width = type->get_width();
    if (int_encoding)
      return mk_sort(SMT_SORT_REAL);
    else
      return mk_sort(SMT_SORT_BV, width);
  }
  case type2t::string_id:
  {
    smt_sort *d = (int_encoding)? mk_sort(SMT_SORT_INT)
                                : mk_sort(SMT_SORT_BV, config.ansi_c.int_width);
    smt_sort *r = (int_encoding)? mk_sort(SMT_SORT_INT)
                                : mk_sort(SMT_SORT_BV, 8);
    return mk_sort(SMT_SORT_ARRAY, d, r);
  }
  case type2t::array_id:
  {
    // All arrays are indexed by integerse
    smt_sort *d = (int_encoding)? mk_sort(SMT_SORT_INT)
                                : mk_sort(SMT_SORT_BV, config.ansi_c.int_width);
    const array_type2t &arr = to_array_type(type);
    smt_sort *r = convert_sort(arr.subtype);
    return mk_sort(SMT_SORT_ARRAY, d, r);
  }
  case type2t::code_id:
  case type2t::cpp_name_id:
  case type2t::symbol_id:
  case type2t::empty_id:
  default:
    assert(0 && "Unexpected type ID reached SMT conversion");
  }
}

smt_ast *
smt_convt::convert_terminal(const expr2tc &expr)
{
  switch (expr->expr_id) {
  case expr2t::constant_int_id:
  {
    bool sign = is_signedbv_type(expr);
    const constant_int2t &theint = to_constant_int2t(expr);
    unsigned int width = expr->type->get_width();
    if (int_encoding)
      return mk_smt_int(theint.constant_value, sign, expr);
    else
      return mk_smt_bvint(theint.constant_value, sign, width, expr);
  }
  case expr2t::constant_fixedbv_id:
  {
    const constant_fixedbv2t &thereal = to_constant_fixedbv2t(expr);
    return mk_smt_real(thereal.value.to_integer(), expr);
  }
  case expr2t::constant_bool_id:
  {
    const constant_bool2t &thebool = to_constant_bool2t(expr);
    return mk_smt_bool(thebool.constant_value, expr);
  }
  case expr2t::symbol_id:
  {
    // Can't do this right now due to not having sort conversion yet.
    const symbol2t &sym = to_symbol2t(expr);
    std::string name = sym.get_symbol_name();
    const smt_sort *sort = convert_sort(sym.type);
    return mk_smt_symbol(name, sort, expr);
  }
  default:
    return mk_func_app(NULL, SMT_FUNC_HACKS, NULL, 0, expr);
  }
}

const smt_convt::expr_op_convert
smt_convt::smt_convert_table[expr2t::end_expr_id] =  {
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //const int
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //const bool
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //const string
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //const struct
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //const union
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //const array
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //const array_of
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //symbol
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //typecast
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //if
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //equality
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //notequal
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //lt
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //gt
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //lte
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //gte
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //not
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //and
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //or
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //xor
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //impiles
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //bitand
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //bitor
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //bitxor
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //bitnand
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //bitnor
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //bitnxor
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //bitnot
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //lshr
{ SMT_FUNC_NEG, SMT_FUNC_BVNEG, SMT_FUNC_BVNEG, 1, SMT_SORT_ALLINTS},  //neg
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //abs
{ SMT_FUNC_ADD, SMT_FUNC_BVADD, SMT_FUNC_BVADD, 2, SMT_SORT_ALLINTS},//add
{ SMT_FUNC_SUB, SMT_FUNC_BVSUB, SMT_FUNC_BVSUB, 2, SMT_SORT_ALLINTS},//sub
{ SMT_FUNC_MUL, SMT_FUNC_BVMUL, SMT_FUNC_BVMUL, 2, SMT_SORT_ALLINTS},//mul
{ SMT_FUNC_DIV, SMT_FUNC_BVDIV, SMT_FUNC_BVDIV, 2, SMT_SORT_ALLINTS},//div
{ SMT_FUNC_MOD, SMT_FUNC_BVSMOD, SMT_FUNC_BVUMOD, 2, SMT_SORT_BV | SMT_SORT_INT},//mod
{ SMT_FUNC_SHL, SMT_FUNC_BVSHL, SMT_FUNC_BVSHL, 2, SMT_SORT_BV | SMT_SORT_INT},  //shl
{ SMT_FUNC_INVALID, SMT_FUNC_BVASHR, SMT_FUNC_BVASHR, 2, SMT_SORT_BV},  //ashr
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //dyn_obj_id
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //same_obj_id
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //ptr_offs
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //ptr_obj
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //addr_of
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //byte_extract
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //byte_update
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //with
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //member
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //index
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //zero_str_id
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //zero_len_str
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //isnan
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //overflow
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //overflow_cast
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //overflow_neg
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //unknown
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //invalid
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //null_obj
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //deref
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //valid_obj
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //deallocated
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //dyn_size
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //sideeffect
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_block
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_assign
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_init
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_decl
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_printf
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_expr
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_return
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_skip
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_free
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_goto
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //obj_desc
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_func_call
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_comma
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //invalid_ptr
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //buffer_sz
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_asm
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //cpp_del_arr
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //cpp_del_id
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //cpp_catch
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //cpp_throw
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //cpp_throw_dec
};
