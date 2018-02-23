#include <boolector_conv.h>
#include <cstring>

extern "C" {
#include <btorcore.h>
}

smt_convt *create_new_boolector_solver(
  bool int_encoding,
  const namespacet &ns,
  tuple_iface **tuple_api __attribute__((unused)),
  array_iface **array_api,
  fp_convt **fp_api)
{
  boolector_convt *conv = new boolector_convt(int_encoding, ns);
  *array_api = static_cast<array_iface *>(conv);
  *fp_api = static_cast<fp_convt *>(conv);
  return conv;
}

boolector_convt::boolector_convt(bool int_encoding, const namespacet &ns)
  : smt_convt(int_encoding, ns), array_iface(false, false), fp_convt(this)
{
  if(int_encoding)
  {
    std::cerr << "Boolector does not support integer encoding mode"
              << std::endl;
    abort();
  }

  btor = boolector_new();
  boolector_set_opt(btor, BTOR_OPT_MODEL_GEN, 1);
  boolector_set_opt(btor, BTOR_OPT_AUTO_CLEANUP, 1);
}

boolector_convt::~boolector_convt()
{
  boolector_delete(btor);
  btor = nullptr;
}

smt_convt::resultt boolector_convt::dec_solve()
{
  pre_solve();

  int result = boolector_sat(btor);

  if(result == BOOLECTOR_SAT)
    return P_SATISFIABLE;

  if(result == BOOLECTOR_UNSAT)
    return P_UNSATISFIABLE;

  return P_ERROR;
}

const std::string boolector_convt::solver_text()
{
  std::string ss = "Boolector ";
  ss += btor_version(btor);
  return ss;
}

void boolector_convt::assert_ast(const smt_ast *a)
{
  boolector_assert(btor, to_solver_smt_ast<btor_smt_ast>(a)->a);
}

smt_ast *boolector_convt::mk_func_app(
  const smt_sort *s,
  smt_func_kind k,
  const smt_ast *const *args,
  unsigned int numargs)
{
  const btor_smt_ast *asts[4];
  unsigned int i;

  assert(numargs <= 4);
  for(i = 0; i < numargs; i++)
  {
    asts[i] = to_solver_smt_ast<btor_smt_ast>(args[i]);
    // Structs should never reach the SMT solver
    assert(asts[i]->sort->id != SMT_SORT_STRUCT);
  }

  switch(k)
  {
  case SMT_FUNC_BVADD:
    return new_ast(s, boolector_add(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVSUB:
    return new_ast(s, boolector_sub(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVMUL:
    return new_ast(s, boolector_mul(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVSMOD:
    return new_ast(s, boolector_srem(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVUMOD:
    return new_ast(s, boolector_urem(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVSDIV:
    return new_ast(s, boolector_sdiv(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVUDIV:
    return new_ast(s, boolector_udiv(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVSHL:
    return fix_up_shift(boolector_sll, asts[0], asts[1], s);
  case SMT_FUNC_BVLSHR:
    return fix_up_shift(boolector_srl, asts[0], asts[1], s);
  case SMT_FUNC_BVASHR:
    return fix_up_shift(boolector_sra, asts[0], asts[1], s);
  case SMT_FUNC_BVNEG:
    return new_ast(s, boolector_neg(btor, asts[0]->a));
  case SMT_FUNC_BVNOT:
  case SMT_FUNC_NOT:
    return new_ast(s, boolector_not(btor, asts[0]->a));
  case SMT_FUNC_BVNXOR:
    return new_ast(s, boolector_xnor(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVNOR:
    return new_ast(s, boolector_nor(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVNAND:
    return new_ast(s, boolector_nand(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVXOR:
  case SMT_FUNC_XOR:
    return new_ast(s, boolector_xor(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVOR:
  case SMT_FUNC_OR:
    return new_ast(s, boolector_or(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVAND:
  case SMT_FUNC_AND:
    return new_ast(s, boolector_and(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_IMPLIES:
    return new_ast(s, boolector_implies(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVULT:
    return new_ast(s, boolector_ult(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVSLT:
    return new_ast(s, boolector_slt(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVULTE:
    return new_ast(s, boolector_ulte(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVSLTE:
    return new_ast(s, boolector_slte(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVUGT:
    return new_ast(s, boolector_ugt(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVSGT:
    return new_ast(s, boolector_sgt(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVUGTE:
    return new_ast(s, boolector_ugte(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_BVSGTE:
    return new_ast(s, boolector_sgte(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_EQ:
    return new_ast(s, boolector_eq(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_NOTEQ:
    return new_ast(s, boolector_ne(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_ITE:
    return new_ast(s, boolector_cond(btor, asts[0]->a, asts[1]->a, asts[2]->a));
  case SMT_FUNC_STORE:
    return new_ast(
      s, boolector_write(btor, asts[0]->a, asts[1]->a, asts[2]->a));
  case SMT_FUNC_SELECT:
    return new_ast(s, boolector_read(btor, asts[0]->a, asts[1]->a));
  case SMT_FUNC_CONCAT:
    return new_ast(s, boolector_concat(btor, asts[0]->a, asts[1]->a));
  default:
    std::cerr << "Unhandled SMT func \"" << smt_func_name_table[k]
              << "\" in boolector conv" << std::endl;
    abort();
  }
}

smt_ast *boolector_convt::mk_smt_int(
  const mp_integer &theint __attribute__((unused)),
  bool sign __attribute__((unused)))
{
  std::cerr << "Boolector can't create integer sorts" << std::endl;
  abort();
}

smt_ast *boolector_convt::mk_smt_real(const std::string &str
                                      __attribute__((unused)))
{
  std::cerr << "Boolector can't create Real sorts" << std::endl;
  abort();
}

smt_astt boolector_convt::mk_smt_bv(smt_sortt s, const mp_integer &theint)
{
  std::size_t w = s->get_data_width();

  if(w > 32)
  {
    // We have to pass things around via means of strings, becausae boolector
    // uses native int types as arguments to its functions, rather than fixed
    // width integers. Seeing how amd64 is LP64, there's no way to pump 64 bit
    // ints to boolector natively.
    if(w > 64)
    {
      std::cerr << "Boolector backend assumes maximum bitwidth is 64, sorry"
                << std::endl;
      abort();
    }

    char buffer[65];
    memset(buffer, 0, sizeof(buffer));

    // Note that boolector has the most significant bit first in bit strings.
    int64_t num = theint.to_int64();
    uint64_t bit = 1ULL << (w - 1);
    for(unsigned int i = 0; i < w; i++)
    {
      if(num & bit)
        buffer[i] = '1';
      else
        buffer[i] = '0';

      bit >>= 1;
    }

    BoolectorNode *node = boolector_const(btor, buffer);
    return new_ast(s, node);
  }

  BoolectorNode *node;
  if(s->id == SMT_SORT_SBV)
  {
    node = boolector_int(
      btor, theint.to_long(), to_solver_smt_sort<BoolectorSort>(s)->s);
  }
  else
  {
    node = boolector_unsigned_int(
      btor, theint.to_ulong(), to_solver_smt_sort<BoolectorSort>(s)->s);
  }

  return new_ast(s, node);
}

smt_ast *boolector_convt::mk_smt_bool(bool val)
{
  BoolectorNode *node = (val) ? boolector_true(btor) : boolector_false(btor);
  const smt_sort *sort = boolean_sort;
  return new_ast(sort, node);
}

smt_ast *boolector_convt::mk_array_symbol(
  const std::string &name,
  const smt_sort *s,
  smt_sortt array_subtype __attribute__((unused)))
{
  return mk_smt_symbol(name, s);
}

smt_ast *
boolector_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  symtable_type::iterator it = symtable.find(name);
  if(it != symtable.end())
    return it->second;

  BoolectorNode *node;

  switch(s->id)
  {
  case SMT_SORT_SBV:
  case SMT_SORT_UBV:
  case SMT_SORT_FIXEDBV:
  case SMT_SORT_FAKE_FLOATBV:
  case SMT_SORT_FAKE_FLOATBV_RM:
    node = boolector_var(
      btor, to_solver_smt_sort<BoolectorSort>(s)->s, name.c_str());
    break;

  case SMT_SORT_BOOL:
    node = boolector_var(
      btor, to_solver_smt_sort<BoolectorSort>(s)->s, name.c_str());
    break;

  case SMT_SORT_ARRAY:
    node = boolector_array(
      btor, to_solver_smt_sort<BoolectorSort>(s)->s, name.c_str());
    break;

  default:
    std::cerr << "Unknown type for symbol\n";
    abort();
  }

  btor_smt_ast *ast = new_ast(s, node);

  symtable.insert(symtable_type::value_type(name, ast));
  return ast;
}

smt_ast *boolector_convt::mk_extract(
  const smt_ast *a,
  unsigned int high,
  unsigned int low,
  const smt_sort *s)
{
  const btor_smt_ast *ast = to_solver_smt_ast<btor_smt_ast>(a);
  BoolectorNode *b = boolector_slice(btor, ast->a, high, low);
  return new_ast(s, b);
}

bool boolector_convt::get_bool(const smt_ast *a)
{
  const btor_smt_ast *ast = to_solver_smt_ast<btor_smt_ast>(a);
  const char *result = boolector_bv_assignment(btor, ast->a);

  assert(result != NULL && "Boolector returned null bv assignment string");

  bool res;
  switch(*result)
  {
  case '1':
    res = true;
    break;
  case '0':
    res = false;
    break;
  default:
    abort();
  }

  boolector_free_bv_assignment(btor, result);
  return res;
}

BigInt boolector_convt::get_bv(smt_astt a)
{
  const btor_smt_ast *ast = to_solver_smt_ast<btor_smt_ast>(a);

  const char *result = boolector_bv_assignment(btor, ast->a);
  BigInt val = string2integer(result, 2);
  boolector_free_bv_assignment(btor, result);

  return val;
}

expr2tc boolector_convt::get_array_elem(
  const smt_ast *array,
  uint64_t index,
  const type2tc &subtype)
{
  const btor_smt_ast *ast = to_solver_smt_ast<btor_smt_ast>(array);

  int size;
  char **indicies, **values;
  boolector_array_assignment(btor, ast->a, &indicies, &values, &size);

  BigInt val = 0;
  if(size > 0)
  {
    for(int i = 0; i < size; i++)
    {
      auto idx = string2integer(indicies[i], 2);
      if(idx.to_uint64() == index)
      {
        val = string2integer(values[i], 2);
        break;
      }
    }

    boolector_free_array_assignment(btor, indicies, values, size);
    return build_bv(subtype, val);
  }

  return gen_zero(subtype);
}

const smt_ast *boolector_convt::overflow_arith(const expr2tc &expr)
{
  const overflow2t &overflow = to_overflow2t(expr);
  const arith_2ops &opers = static_cast<const arith_2ops &>(*overflow.operand);

  const btor_smt_ast *side1 =
    to_solver_smt_ast<btor_smt_ast>(convert_ast(opers.side_1));
  const btor_smt_ast *side2 =
    to_solver_smt_ast<btor_smt_ast>(convert_ast(opers.side_2));

  // Guess whether we're performing a signed or unsigned comparison.
  bool is_signed =
    (is_signedbv_type(opers.side_1) || is_signedbv_type(opers.side_2));

  BoolectorNode *res;
  if(is_add2t(overflow.operand))
  {
    if(is_signed)
    {
      res = boolector_saddo(btor, side1->a, side2->a);
    }
    else
    {
      res = boolector_uaddo(btor, side1->a, side2->a);
    }
  }
  else if(is_sub2t(overflow.operand))
  {
    if(is_signed)
    {
      res = boolector_ssubo(btor, side1->a, side2->a);
    }
    else
    {
      res = boolector_usubo(btor, side1->a, side2->a);
    }
  }
  else if(is_mul2t(overflow.operand))
  {
    if(is_signed)
    {
      res = boolector_smulo(btor, side1->a, side2->a);
    }
    else
    {
      res = boolector_umulo(btor, side1->a, side2->a);
    }
  }
  else if(is_div2t(overflow.operand) || is_modulus2t(overflow.operand))
  {
    res = boolector_sdivo(btor, side1->a, side2->a);
  }
  else
  {
    return smt_convt::overflow_arith(expr);
  }

  const smt_sort *s = boolean_sort;
  return new_ast(s, res);
}

const smt_ast *
boolector_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  return default_convert_array_of(init_val, domain_width, this);
}

void boolector_convt::add_array_constraints_for_solving()
{
}

void boolector_convt::push_array_ctx()
{
}

void boolector_convt::pop_array_ctx()
{
}

smt_ast *boolector_convt::fix_up_shift(
  shift_func_ptr fptr,
  const btor_smt_ast *op0,
  const btor_smt_ast *op1,
  smt_sortt res_sort)
{
  BoolectorNode *data_op, *shift_amount;
  bool need_to_shift_down = false;
  unsigned int bwidth;

  data_op = op0->a;
  bwidth = log2(op0->sort->get_data_width());

  // If we're a non-power-of-x number, some zero extension has to occur
  if(pow(2.0, bwidth) < op0->sort->get_data_width())
  {
    // Zero extend up to bwidth + 1
    bwidth++;
    unsigned int new_size = pow(2.0, bwidth);
    unsigned int diff = new_size - op0->sort->get_data_width();
    smt_sortt newsort = mk_int_bv_sort(SMT_SORT_UBV, new_size);
    smt_astt zeroext = convert_zero_ext(op0, newsort, diff);
    data_op = to_solver_smt_ast<btor_smt_ast>(zeroext)->a;
    need_to_shift_down = true;
  }

  // We also need to reduce the shift-amount operand down to log2(data_op) len
  shift_amount = boolector_slice(btor, op1->a, bwidth - 1, 0);

  BoolectorNode *shift = fptr(btor, data_op, shift_amount);

  // If zero extension occurred, cut off the top few bits of this value.
  if(need_to_shift_down)
    shift = boolector_slice(btor, shift, res_sort->get_data_width() - 1, 0);

  return new_ast(res_sort, shift);
}

void boolector_convt::dump_smt()
{
  boolector_dump_smt2(btor, stdout);
}

void btor_smt_ast::dump() const
{
  boolector_dump_smt2_node(boolector_get_btor(a), stdout, a);
}

void boolector_convt::print_model()
{
  boolector_print_model(btor, const_cast<char *>("smt2"), stdout);
}

smt_sortt boolector_convt::mk_bool_sort()
{
  return new solver_smt_sort<BoolectorSort>(
    SMT_SORT_BOOL, boolector_bool_sort(btor), 1);
}

smt_sortt boolector_convt::mk_bv_sort(const smt_sort_kind k, std::size_t width)
{
  return new solver_smt_sort<BoolectorSort>(
    k, boolector_bitvec_sort(btor, width), width);
}

smt_sortt boolector_convt::mk_array_sort(smt_sortt domain, smt_sortt range)
{
  auto domain_sort = to_solver_smt_sort<BoolectorSort>(domain);
  auto range_sort = to_solver_smt_sort<BoolectorSort>(range);

  auto t = boolector_array_sort(btor, domain_sort->s, range_sort->s);
  return new solver_smt_sort<BoolectorSort>(
    SMT_SORT_ARRAY, t, domain_sort->get_data_width(), range);
}

smt_sortt boolector_convt::mk_bv_fp_sort(std::size_t ew, std::size_t sw)
{
  return new solver_smt_sort<BoolectorSort>(
    SMT_SORT_FAKE_FLOATBV,
    boolector_bitvec_sort(btor, ew + sw + 1),
    ew + sw + 1,
    sw);
}

smt_sortt boolector_convt::mk_bv_fp_rm_sort()
{
  return new solver_smt_sort<BoolectorSort>(
    SMT_SORT_FAKE_FLOATBV_RM, boolector_bitvec_sort(btor, 2), 2);
}
