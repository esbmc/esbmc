/*******************************************************************\
Module: Unit tests for src/llm/goto: GOTO renderer and expression parser
\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2_utils.h>
#include <llm/goto/parse_expr.h>
#include <llm/goto/render.h>
#include <llm/llm_client.h>
#include <util/config/config.h>
#include <util/lang/c_types.h>
#include <util/symtab/context.h>

#include <algorithm>

namespace
{
struct fixturet
{
  contextt context;
  namespacet ns{context};
  type2tc int_t, uint_t, uchar_t, ptr_t, struct_t, struct_ptr_t;
  expr2tc n, a, p, c, s, i, global_s;
  goto_functiont function;
  goto_programt::targett head;

  fixturet()
  {
    config.ansi_c.set_data_model(configt::LP64);
    int_t = int_type2();
    uint_t = uint_type2();
    uchar_t = unsignedbv_type2tc(8);
    ptr_t = pointer_type2tc(int_t);
    struct_t = struct_type2tc(
      std::vector<type2tc>{int_t, ptr_t},
      std::vector<irep_idt>{"len", "data"},
      std::vector<irep_idt>{"len", "data"},
      "tag-struct S");
    struct_ptr_t = pointer_type2tc(struct_t);

    n = variable("c:@F@sum@n", "n", uint_t);
    a = variable("c:@F@sum@a", "a", ptr_t);
    p = variable("c:@F@sum@p", "p", struct_ptr_t);
    c = variable("c:@F@sum@c", "c", uchar_t);
    s = variable("c:@F@sum@s", "s", int_t);
    i = variable("c:@F@sum@i", "i", uint_t);
    global_s = variable("c:@s", "s", int_t);
    variable(
      "c:@F@sum",
      "sum",
      code_type2tc(
        std::vector<type2tc>{}, int_t, std::vector<irep_idt>{}, false));

    function.type = code_type2tc(
      std::vector<type2tc>{uint_t, ptr_t, struct_ptr_t, uchar_t},
      int_t,
      std::vector<irep_idt>{
        "c:@F@sum@n", "c:@F@sum@a", "c:@F@sum@p", "c:@F@sum@c"},
      false);
    build_body();
  }

  expr2tc variable(const irep_idt &id, const irep_idt &name, const type2tc &t)
  {
    symbolt symbol;
    symbol.id = id;
    symbol.name = name;
    symbol.mode = "C";
    symbol.lvalue = true;
    set_symbol_type(symbol, t);
    context.add(symbol);
    return symbol2tc(t, id);
  }

  void build_body()
  {
    goto_programt &body = function.body;
    auto add = [&](goto_program_instruction_typet type) {
      return body.add_instruction(type);
    };
    auto decl = [&](const expr2tc &sym) {
      add(DECL)->code = code_decl2tc(sym->type, to_symbol2t(sym).thename);
    };
    auto assign = [&](const expr2tc &lhs, const expr2tc &rhs) {
      add(ASSIGN)->code = code_assign2tc(lhs, rhs);
    };

    decl(s);
    assign(s, gen_zero(int_t));
    decl(i);
    assign(i, gen_zero(uint_t));
    head = add(SKIP);
    auto test = add(GOTO);
    test->guard = not2tc(lessthan2tc(i, n));
    assign(s, add2tc(int_t, s, index2tc(int_t, a, i)));
    assign(i, add2tc(uint_t, i, gen_one(uint_t)));
    auto latch = add(GOTO);
    latch->targets.push_back(head);
    auto ret = add(RETURN);
    ret->code = code_return2tc(add2tc(int_t, s, global_s));
    test->targets.push_back(ret);
    add(END_FUNCTION);
    body.update();
  }
};

std::optional<expr2tc> parse(const fixturet &f, const std::string &text)
{
  const llm::scopet scope =
    llm::render_function("c:@F@sum", f.function, f.ns).scope;
  return llm::parse_expr(text, scope, f.ns);
}
} // namespace

TEST_CASE("renderer prints a while loop as pseudo-C", "[llm]")
{
  fixturet f;
  const llm::rendered_functiont r =
    llm::render_function("c:@F@sum", f.function, f.ns);

  CHECK(
    r.text ==
    "int sum(unsigned int n, int *a, struct S *p, unsigned char c)\n"
    "{\n"
    "  int s;\n"
    "  s = 0;\n"
    "  unsigned int i;\n"
    "  i = 0u;\n"
    "  /* L1 modifies: s, i */\n"
    "  /* L1 */ while (i < n)\n"
    "  {\n"
    "    s = s + a[i];\n"
    "    i = i + 1u;\n"
    "  }\n"
    "  return s + s_2;\n"
    "}\n");
  REQUIRE(r.loops.size() == 1);
  CHECK(r.loops.at("L1") == goto_programt::const_targett(f.head));
  CHECK(r.scope.at("s") == f.s);
  CHECK(r.scope.at("s_2") == f.global_s);
  CHECK(r.scope.count("sum") == 0);
}

TEST_CASE("renderer recovers if/else, break, continue and do-while", "[llm]")
{
  fixturet f;
  const expr2tc &x = f.s;
  const expr2tc one = gen_one(f.int_t);
  auto num = [&](int v) { return constant_int2tc(f.int_t, BigInt(v)); };

  goto_functiont g;
  g.type = code_type2tc(
    std::vector<type2tc>{f.int_t},
    f.int_t,
    std::vector<irep_idt>{"c:@F@sum@s"},
    false);
  goto_programt &body = g.body;
  auto jump = [&](const expr2tc &guard) {
    auto i = body.add_instruction(GOTO);
    i->guard = guard;
    return i;
  };
  auto assign = [&](const expr2tc &rhs) {
    body.add_instruction(ASSIGN)->code = code_assign2tc(x, rhs);
  };

  auto head = jump(not2tc(greaterthan2tc(x, gen_zero(f.int_t))));
  assign(sub2tc(f.int_t, x, one));
  auto skip_else = jump(gen_true_expr());
  auto else_part = body.add_instruction(ASSIGN);
  else_part->code = code_assign2tc(x, add2tc(f.int_t, x, num(2)));
  auto brk = jump(equality2tc(x, num(5)));
  auto next_iteration = jump(equality2tc(x, num(6)));
  jump(equality2tc(x, num(7)))->targets.push_back(head);
  auto latch = jump(lessthan2tc(x, num(10)));
  latch->targets.push_back(head);
  next_iteration->targets.push_back(latch);
  auto forward = jump(gen_true_expr());
  assign(gen_zero(f.int_t));
  auto ret = body.add_instruction(RETURN);
  ret->code = code_return2tc(x);
  auto to_end = jump(gen_true_expr());
  auto end = body.add_instruction(END_FUNCTION);

  head->targets.push_back(else_part);
  skip_else->targets.push_back(brk);
  brk->targets.push_back(forward);
  forward->targets.push_back(to_end);
  to_end->targets.push_back(end);
  body.update();

  CHECK(
    llm::render_function("c:@F@f", g, f.ns).text ==
    "int f(int s)\n"
    "{\n"
    "  /* L1 modifies: s */\n"
    "  /* L1 */ do\n"
    "  {\n"
    "    label_1:\n"
    "    if (s > 0)\n"
    "    {\n"
    "      s = s - 1;\n"
    "    }\n"
    "    else\n"
    "    {\n"
    "      s = s + 2;\n"
    "    }\n"
    "    if (s == 5) break;\n"
    "    if (s == 6) continue;\n"
    "    if (s == 7) goto label_1;\n"
    "  } while (s < 10);\n"
    "  goto label_2;\n"
    "  s = 0;\n"
    "  return s;\n"
    "  label_2:\n"
    "  return;\n"
    "}\n");
}

TEST_CASE("parser builds the expected IRep2", "[llm]")
{
  fixturet f;
  const expr2tc zero = gen_zero(f.int_t);

  CHECK(
    *parse(f, "i <= n && s >= 0") ==
    and2tc(lessthanequal2tc(f.i, f.n), greaterthanequal2tc(f.s, zero)));
  CHECK(
    *parse(f, "a[i] == *a") ==
    equality2tc(index2tc(f.int_t, f.a, f.i), dereference2tc(f.int_t, f.a)));
  CHECK(
    *parse(f, "p->len > 0") ==
    greaterthan2tc(
      member2tc(f.int_t, dereference2tc(f.struct_t, f.p), "len"), zero));
  CHECK(*parse(f, "a != NULL") == notequal2tc(f.a, gen_zero(f.ptr_t)));
  CHECK(*parse(f, "!s") == not2tc(notequal2tc(f.s, zero)));
  CHECK(
    *parse(f, "s_2 > 0 ? s : -s") ==
    if2tc(
      f.int_t, greaterthan2tc(f.global_s, zero), f.s, neg2tc(f.int_t, f.s)));
  CHECK(*parse(f, "true") == gen_true_expr());
}

TEST_CASE("parser applies C integer conversions", "[llm]")
{
  fixturet f;
  const expr2tc one = gen_one(f.int_t);

  CHECK(*parse(f, "c + 1") == add2tc(f.int_t, typecast2tc(f.int_t, f.c), one));
  CHECK(*parse(f, "i < s") == lessthan2tc(f.i, typecast2tc(f.uint_t, f.s)));
  CHECK(*parse(f, "(unsigned char)s") == typecast2tc(f.uchar_t, f.s));
  CHECK(*parse(f, "(int)s") == f.s);
  CHECK(*parse(f, "s >> 1") == ashr2tc(f.int_t, f.s, one));
  CHECK(*parse(f, "i >> 1") == lshr2tc(f.uint_t, f.i, one));
}

TEST_CASE("parser types integer literals as C does", "[llm]")
{
  fixturet f;
  auto type_of = [&](const std::string &text) {
    return (*parse(f, text))->type;
  };

  CHECK(type_of("2147483647") == f.int_t);
  CHECK(type_of("2147483648") == long_int_type2());
  CHECK(type_of("0xFFFFFFFF") == f.uint_t);
  CHECK(type_of("1u") == f.uint_t);
  CHECK(type_of("1ul") == long_uint_type2());
  CHECK(type_of("1LL") == long_long_int_type2());
  CHECK(*parse(f, "-1") == constant_int2tc(f.int_t, BigInt(-1)));
  CHECK(
    *parse(f, "-1u") == neg2tc(f.uint_t, constant_int2tc(f.uint_t, BigInt(1))));
  CHECK(*parse(f, "010") == constant_int2tc(f.int_t, BigInt(8)));
}

TEST_CASE("parser rejects malformed or ill-typed input", "[llm]")
{
  fixturet f;
  for (const char *text :
       {"x < 1",
        "i <",
        "(s",
        "s @ 1",
        "s.len",
        "p.len",
        "p->size",
        "a + a",
        "a < s",
        "1.5 < s",
        "*s",
        "s[0]",
        "1uu",
        "0x",
        "99999999999999999999999",
        "(float)s",
        "(unsigned signed)s",
        "1lul",
        "1lL",
        "&1",
        "& &s",
        "&(s + 1)",
        ""})
  {
    INFO(text);
    CHECK_FALSE(parse(f, text).has_value());
  }
}

TEST_CASE("printing then parsing gives back the expression", "[llm]")
{
  fixturet f;
  const llm::scopet scope =
    llm::render_function("c:@F@sum", f.function, f.ns).scope;
  const expr2tc zero = gen_zero(f.int_t);

  for (const expr2tc &e :
       {and2tc(lessthanequal2tc(f.i, f.n), greaterthanequal2tc(f.s, zero)),
        equality2tc(
          f.s,
          div2tc(
            f.int_t,
            mul2tc(f.int_t, f.s, sub2tc(f.int_t, f.s, gen_one(f.int_t))),
            constant_int2tc(f.int_t, BigInt(2)))),
        lessthan2tc(f.i, typecast2tc(f.uint_t, f.s)),
        or2tc(
          equality2tc(f.global_s, zero),
          greaterthan2tc(index2tc(f.int_t, f.a, f.i), f.s)),
        notequal2tc(
          member2tc(f.ptr_t, dereference2tc(f.struct_t, f.p), "data"),
          gen_zero(f.ptr_t)),
        lessthan2tc(
          add2tc(f.int_t, f.s, constant_int2tc(f.int_t, -BigInt::power2(31))),
          zero),
        equality2tc(
          typecast2tc(long_int_type2(), f.s),
          constant_int2tc(long_int_type2(), -BigInt::power2(63)))})
  {
    const std::string text = llm::render_expr(e, scope, f.ns);
    INFO(text);
    const std::optional<expr2tc> back = llm::parse_expr(text, scope, f.ns);
    REQUIRE(back.has_value());
    CHECK(*back == e);
  }
}

TEST_CASE("renderer spells the C type of every declaration", "[llm]")
{
  fixturet f;
  goto_functiont g;
  g.type = code_type2tc(
    std::vector<type2tc>{get_bool_type()},
    get_empty_type(),
    std::vector<irep_idt>{"c:@F@g@b"},
    true);
  f.variable("c:@F@g@b", "b", get_bool_type());

  const std::vector<std::pair<const char *, type2tc>> decls = {
    {"$h", signedbv_type2tc(16)},
    {"2l", long_uint_type2()},
    {"w", unsignedbv_type2tc(24)},
    {"x", float_type2()},
    {"y", double_type2()},
    {"z", floatbv_type2tc(112, 15)},
    {"u",
     union_type2tc(
       std::vector<type2tc>{f.int_t},
       std::vector<irep_idt>{"m"},
       std::vector<irep_idt>{"m"},
       "tag-union U")},
    {"t", symbol_type2tc("tag-struct T")},
    {"v", pointer_type2tc(get_empty_type())},
    {"arr", array_type2tc(f.int_t, constant_int2tc(size_type2(), 4), false)},
    {"open", array_type2tc(f.int_t, expr2tc(), true)},
    {"vec", vector_type2tc(f.int_t, constant_int2tc(size_type2(), 4))}};
  for (const auto &[name, type] : decls)
  {
    const std::string id = std::string("c:@F@g@") + name;
    f.variable(id, name, type);
    g.body.add_instruction(DECL)->code = code_decl2tc(type, id);
  }
  const expr2tc l = symbol2tc(long_uint_type2(), "c:@F@g@2l");
  g.body.add_instruction(ASSIGN)->code =
    code_assign2tc(l, constant_int2tc(long_uint_type2(), BigInt(5)));
  g.body.add_instruction(DECL)->code = code_decl2tc(
    f.uchar_t, "c:@F@sum@c", constant_int2tc(f.uchar_t, BigInt(7)));
  g.body.add_instruction(END_FUNCTION);
  g.body.update();

  CHECK(
    llm::render_function("c:@F@g", g, f.ns).text ==
    "void g(_Bool b, ...)\n"
    "{\n"
    "  short _h;\n"
    "  unsigned long _2l;\n"
    "  unsigned _BitInt(24) w;\n"
    "  float x;\n"
    "  double y;\n"
    "  long double z;\n"
    "  union U u;\n"
    "  struct T t;\n"
    "  void *v;\n"
    "  int arr[4ul];\n"
    "  int open[];\n"
    "  signed int vector [4ul] vec;\n"
    "  _2l = 5ul;\n"
    "  unsigned char c = 7;\n"
    "}\n");
}

TEST_CASE("renderer spells long long under ILP32", "[llm]")
{
  fixturet f;
  config.ansi_c.set_data_model(configt::ILP32);
  const llm::scopet scope;
  CHECK(
    llm::render_expr(
      constant_int2tc(long_long_int_type2(), BigInt(5)), scope, f.ns) == "5ll");
  CHECK(
    llm::render_expr(
      constant_int2tc(long_long_uint_type2(), BigInt(5)), scope, f.ns) ==
    "5ull");

  goto_functiont g;
  g.type = code_type2tc(
    std::vector<type2tc>{long_long_int_type2()},
    f.int_t,
    std::vector<irep_idt>{"c:@F@sum@n"},
    false);
  g.body.add_instruction(END_FUNCTION);
  g.body.update();
  CHECK(
    llm::render_function("c:@F@sum", g, f.ns).text ==
    "int sum(long long n)\n{\n}\n");
  config.ansi_c.set_data_model(configt::LP64);
}

TEST_CASE(
  "renderer prints every instruction kind inside an infinite loop",
  "[llm]")
{
  fixturet f;
  const expr2tc &x = f.s;
  auto num = [&](int v) { return constant_int2tc(f.int_t, BigInt(v)); };
  const expr2tc t = f.variable("c:@F@h@t", "t", f.int_t);
  const expr2tc sum = symbol2tc(
    code_type2tc(
      std::vector<type2tc>{}, f.int_t, std::vector<irep_idt>{}, false),
    "c:@F@sum");

  goto_functiont g;
  g.type = code_type2tc(
    std::vector<type2tc>{f.ptr_t, f.struct_ptr_t},
    get_empty_type(),
    std::vector<irep_idt>{"c:@F@sum@a", "c:@F@sum@p"},
    false);
  goto_programt &body = g.body;
  auto add = [&](goto_program_instruction_typet type) {
    return body.add_instruction(type);
  };

  add(ASSUME)->guard = greaterthan2tc(x, num(0));
  auto head = add(SKIP);
  add(DECL)->code = code_decl2tc(f.int_t, "c:@F@h@t");
  add(ASSIGN)->code = code_assign2tc(t, num(0));
  add(ASSIGN)->code = code_assign2tc(index2tc(f.int_t, f.a, x), t);
  add(ASSIGN)->code = code_assign2tc(
    member2tc(f.int_t, dereference2tc(f.struct_t, f.p), "len"), num(1));
  add(ASSIGN)->code = code_assign2tc(f.p, f.p);
  const expr2tc arr = f.variable(
    "c:@F@h@arr",
    "arr",
    array_type2tc(f.int_t, constant_int2tc(size_type2(), BigInt(2)), false));
  add(ASSIGN)->code = code_assign2tc(index2tc(f.int_t, arr, x), t);
  add(FUNCTION_CALL)->code =
    code_function_call2tc(expr2tc(), sum, std::vector<expr2tc>{x, t});
  add(FUNCTION_CALL)->code =
    code_function_call2tc(x, sum, std::vector<expr2tc>{});
  auto skip_then = add(GOTO);
  skip_then->guard = not2tc(greaterthan2tc(x, num(1)));
  add(ASSIGN)->code = code_assign2tc(x, num(2));
  auto inv = add(LOOP_INVARIANT);
  inv->add_loop_invariant(greaterthanequal2tc(x, num(0)));
  inv->add_loop_invariant(notequal2tc(f.a, gen_zero(f.ptr_t)));
  inv->add_loop_assigns_target(x);
  inv->add_loop_assigns_target(f.p);
  skip_then->targets.push_back(inv);
  add(ATOMIC_BEGIN);
  add(ATOMIC_END);
  add(THROW);
  add(CATCH);
  add(OTHER)->code = code_free2tc(f.a);
  add(OTHER)->code = code_expression2tc(add2tc(f.int_t, x, num(1)));
  add(GOTO)->targets.push_back(head);
  add(ASSERT)->guard = equality2tc(x, num(0));
  add(END_FUNCTION);
  body.update();

  CHECK(
    llm::render_function("c:@F@h", g, f.ns).text ==
    "void h(int *a, struct S *p)\n"
    "{\n"
    "  assume(s > 0);\n"
    "  /* L1 modifies: *a, *p, p, arr, s */\n"
    "  /* L1 */ while (1)\n"
    "  {\n"
    "    int t;\n"
    "    t = 0;\n"
    "    a[s] = t;\n"
    "    p->len = 1;\n"
    "    p = p;\n"
    "    arr[s] = t;\n"
    "    sum(s, t);\n"
    "    s = sum();\n"
    "    if (s > 1)\n"
    "    {\n"
    "      s = 2;\n"
    "    }\n"
    "    invariant(s >= 0); invariant(a != 0); assigns(s, p);\n"
    "    atomic_begin();\n"
    "    atomic_end();\n"
    "    throw;\n"
    "    catch;\n"
    "    FREE(a);\n"
    "    s + 1;\n"
    "  }\n"
    "  assert(s == 0);\n"
    "}\n");
}

TEST_CASE("parser accepts the remaining operators and casts", "[llm]")
{
  fixturet f;
  const expr2tc one = gen_one(f.int_t);
  const type2tc array_t =
    array_type2tc(f.int_t, constant_int2tc(size_type2(), 4), false);
  llm::scopet scope = llm::render_function("c:@F@sum", f.function, f.ns).scope;
  const expr2tc arr = symbol2tc(array_t, "arr");
  scope.emplace("arr", arr);
  auto parse_in = [&](const std::string &text) {
    return llm::parse_expr(text, scope, f.ns);
  };

  CHECK(*parse_in("+c") == typecast2tc(f.int_t, f.c));
  CHECK(*parse_in("~s") == bitnot2tc(f.int_t, f.s));
  CHECK(*parse_in("&s") == address_of2tc(f.int_t, f.s));
  CHECK(*parse_in("false") == gen_false_expr());
  CHECK(*parse_in("arr[1]") == index2tc(f.int_t, arr, one));
  CHECK(*parse_in("NULL == a") == equality2tc(gen_zero(f.ptr_t), f.a));
  CHECK(*parse_in("s << 1") == shl2tc(f.int_t, f.s, one));
  CHECK(
    *parse_in("s % 2") ==
    modulus2tc(f.int_t, f.s, constant_int2tc(f.int_t, BigInt(2))));
  CHECK(*parse_in("s & 1") == bitand2tc(f.int_t, f.s, one));
  CHECK(*parse_in("s | 1") == bitor2tc(f.int_t, f.s, one));
  CHECK(*parse_in("s ^ 1") == bitxor2tc(f.int_t, f.s, one));
  CHECK(*parse_in("1 + a") == add2tc(f.ptr_t, f.a, one));
}

TEST_CASE("parser maps every C integer type name in a cast", "[llm]")
{
  fixturet f;
  auto cast_type = [&](const std::string &type) {
    const std::optional<expr2tc> e = parse(f, "(" + type + ")s");
    REQUIRE(e.has_value());
    REQUIRE(is_typecast2t(*e));
    return (*e)->type;
  };

  CHECK(cast_type("_Bool") == get_bool_type());
  CHECK(cast_type("bool") == get_bool_type());
  CHECK(cast_type("char") == char_type2());
  CHECK(cast_type("signed char") == signedbv_type2tc(8));
  CHECK(cast_type("short") == signedbv_type2tc(16));
  CHECK(cast_type("unsigned short int") == unsignedbv_type2tc(16));
  CHECK(cast_type("long") == long_int_type2());
  CHECK(cast_type("unsigned long") == long_uint_type2());
  CHECK(cast_type("long long int") == long_long_int_type2());
  CHECK(cast_type("unsigned long long") == long_long_uint_type2());
}

TEST_CASE("parser rejects ill-typed operands of every operator", "[llm]")
{
  fixturet f;
  llm::scopet scope = llm::render_function("c:@F@sum", f.function, f.ns).scope;
  scope.emplace("x", symbol2tc(double_type2(), "x"));
  REQUIRE(llm::parse_expr("x > 0", scope, f.ns).has_value());

  for (const char *text :
       {"s s",
        "~x",
        "(int)*p",
        "a[x]",
        "(int int)s",
        "(int foo)s",
        "(unsigned _Bool)s",
        "(char int)s",
        "!*p",
        "-a",
        "s * a",
        "p->1",
        "*p == *p",
        "x << 1",
        "x + 1"})
  {
    INFO(text);
    CHECK_FALSE(llm::parse_expr(text, scope, f.ns).has_value());
  }
}

TEST_CASE("parser survives hostile input", "[llm]")
{
  fixturet f;
  CHECK_FALSE(parse(f, std::string(100000, '(') + "s").has_value());
  CHECK_FALSE(parse(f, std::string(100000, '!') + "s").has_value());
  CHECK(parse(f, std::string(100, '(') + "s" + std::string(100, ')')));

  const llm::scopet scope = {
    {"q", symbol2tc(symbol_type2tc("tag-struct Missing"), "q")}};
  CHECK_FALSE(llm::parse_expr("q.x", scope, f.ns).has_value());
  CHECK_FALSE(llm::parse_expr("q[0]", scope, f.ns).has_value());
}

TEST_CASE("renderer keeps every printed name parsable", "[llm]")
{
  fixturet f;
  const type2tc fn_t = code_type2tc(
    std::vector<type2tc>{f.int_t}, f.int_t, std::vector<irep_idt>{""}, true);
  const type2tc arr_t =
    array_type2tc(f.int_t, constant_int2tc(size_type2(), BigInt(4)), false);
  const expr2tc keyword = f.variable("c:@F@g@true", "true", f.int_t);
  const expr2tc fp = f.variable("c:@F@g@fp", "fp", pointer_type2tc(fn_t));
  const expr2tc pa = f.variable("c:@F@g@pa", "pa", pointer_type2tc(arr_t));

  goto_functiont g;
  g.type = code_type2tc(
    std::vector<type2tc>{f.int_t, f.int_t, fp->type, pa->type},
    f.int_t,
    std::vector<irep_idt>{"c:@F@g@true", "", "c:@F@g@fp", "c:@F@g@pa"},
    false);
  auto branch = g.body.add_instruction(GOTO);
  branch->guard = equality2tc(keyword, gen_zero(f.int_t));
  auto ret = g.body.add_instruction(RETURN);
  ret->code = code_return2tc(keyword);
  auto end = g.body.add_instruction(END_FUNCTION);
  branch->targets.push_back(ret);
  branch->targets.push_back(end);
  g.body.update();

  const llm::rendered_functiont r = llm::render_function("c:@F@g", g, f.ns);
  CHECK(
    r.text ==
    "int g(int true_2, int, int (*fp)(int, ...), int (*pa)[4ul])\n"
    "{\n"
    "  if (true_2 == 0) goto {label_1, label_2};\n"
    "  label_1:\n"
    "  return true_2;\n"
    "  label_2:\n"
    "}\n");
  CHECK(*llm::parse_expr("true_2", r.scope, f.ns) == keyword);
  CHECK(*llm::parse_expr("true", r.scope, f.ns) == gen_true_expr());
  CHECK(r.loops.empty());
}

/// Needs `claude` on PATH.
TEST_CASE(
  "claude haiku proposes a parsable loop invariant",
  "[.manual][llm][claude]")
{
  fixturet f;
  const llm::rendered_functiont r =
    llm::render_function("c:@F@sum", f.function, f.ns);

  llm::configt cfg;
  cfg.backend = llm::backendt::cli;
  cfg.executable = "claude";
  cfg.model = "haiku";
  cfg.extra_args = {"-p", "--model", cfg.model};
  cfg.timeout_ms = 120000;
  auto client = llm::make_client(cfg);

  std::string answer = client->complete(
    {{"system",
      "You propose loop invariants for C programs. Answer with one C "
      "expression and nothing else: no prose, no code fences."},
     {"user",
      "Give an invariant that holds at the head of loop L1 on every "
      "iteration. Use only variables of the function.\n\n" +
        r.text}});

  for (std::size_t at; (at = answer.find("```c")) != std::string::npos;)
    answer.erase(at, 4);
  answer.erase(std::remove(answer.begin(), answer.end(), '`'), answer.end());
  INFO(answer);
  CHECK(llm::parse_expr(answer, r.scope, f.ns).has_value());
}
