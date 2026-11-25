/*******************************************************************\
Module: Unit tests for goto2c: structured loops and loop invariants
\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <goto2c/goto2c.h>
#include <irep2/irep2_utils.h>
#include <util/config/config.h>
#include <util/lang/c_types.h>
#include <util/symtab/context.h>

namespace
{
struct fixturet
{
  contextt context;
  namespacet ns{context};
  type2tc int_t = (config.ansi_c.set_data_model(configt::LP64), int_type2());
  expr2tc n = variable("c:@F@f@n", "n");
  expr2tc i = variable("c:@F@f@i", "i");
  expr2tc t = variable("c:@F@f@t", "t");
  goto_programt body;

  fixturet()
  {
    auto add = [&](goto_program_instruction_typet type) {
      return body.add_instruction(type);
    };
    auto assign = [&](const expr2tc &lhs, const expr2tc &rhs) {
      add(ASSIGN)->code = code_assign2tc(lhs, rhs);
    };
    auto num = [&](int v) { return constant_int2tc(int_t, BigInt(v)); };

    add(DECL)->code = code_decl2tc(int_t, "c:@F@f@i");
    assign(i, num(0));
    auto invariant = add(LOOP_INVARIANT);
    invariant->add_loop_invariant(lessthanequal2tc(i, n));
    invariant->add_loop_assigns_target(i);
    auto head = add(GOTO);
    head->guard = not2tc(lessthan2tc(i, n));
    add(DECL)->code = code_decl2tc(int_t, "c:@F@f@t");
    assign(t, i);
    auto test = add(GOTO);
    test->guard = not2tc(greaterthan2tc(t, num(2)));
    assign(i, add2tc(int_t, i, num(2)));
    auto skip_else = add(GOTO);
    auto otherwise = add(ASSIGN);
    otherwise->code = code_assign2tc(i, add2tc(int_t, i, num(1)));
    auto dead = add(DEAD);
    dead->code = code_dead2tc(int_t, "c:@F@f@t");
    add(GOTO)->targets.push_back(head);
    auto ret = add(RETURN);
    ret->code = code_return2tc(i);
    add(END_FUNCTION);

    head->targets.push_back(ret);
    test->targets.push_back(otherwise);
    skip_else->targets.push_back(dead);
    body.update();
  }

  expr2tc variable(const irep_idt &id, const irep_idt &name)
  {
    symbolt symbol;
    symbol.id = id;
    symbol.name = name;
    symbol.mode = "C";
    symbol.lvalue = true;
    set_symbol_type(symbol, int_t);
    context.add(symbol);
    return symbol2tc(int_t, id);
  }

  std::string translate(bool structured_loops)
  {
    goto2ct goto2c(ns, goto_functionst(), structured_loops);
    goto2c.preprocess(body);
    goto2c.check(body);
    return goto2c.translate(body);
  }
};
} // namespace

TEST_CASE("goto2c prints loops as labels and gotos by default", "[goto2c]")
{
  fixturet f;
  CHECK(
    f.translate(false) ==
    "int i; //  // DECL\n"
    "i=0; //  // ASSIGN\n"
    "__ESBMC_loop_invariant(i <= n); __ESBMC_loop_assigns_impl(&(i)); //  // "
    "LOOP_INVARIANT\n"
    "__ESBMC_goto_label_1:; // Target\n"
    "if(!(i < n)) goto __ESBMC_goto_label_4; //  // GOTO\n"
    "{ // SCOPE BEGIN {1}->{2}\n"
    "int t; //  // DECL\n"
    "t=i; //  // ASSIGN\n"
    "if(!(t > 2)) goto __ESBMC_goto_label_2; //  // GOTO\n"
    "i=i + 2; //  // ASSIGN\n"
    "goto __ESBMC_goto_label_3; //  // GOTO\n"
    "__ESBMC_goto_label_2:; // Target\n"
    "i=i + 1; //  // ASSIGN\n"
    "__ESBMC_goto_label_3:; // Target\n"
    "// dead t;; //  // DEAD\n"
    "} // SCOPE END {2}->{1}\n"
    "goto __ESBMC_goto_label_1; //  // GOTO\n"
    "__ESBMC_goto_label_4:; // Target\n"
    "return i; //  // RETURN\n"
    "// ; //  // END_FUNCTION\n");
}

TEST_CASE("goto2c prints recovered loops as C statements", "[goto2c]")
{
  fixturet f;
  CHECK(
    f.translate(true) ==
    "int i; //  // DECL\n"
    "i=0; //  // ASSIGN\n"
    "__ESBMC_loop_invariant(i <= n); __ESBMC_loop_assigns_impl(&(i)); //  // "
    "LOOP_INVARIANT\n"
    "__ESBMC_goto_label_1:; // Target\n"
    "while(i < n)\n"
    "{\n"
    "{ // SCOPE BEGIN {1}->{2}\n"
    "int t; //  // DECL\n"
    "t=i; //  // ASSIGN\n"
    "if(t > 2)\n"
    "{\n"
    "i=i + 2; //  // ASSIGN\n"
    "}\n"
    "else\n"
    "{\n"
    "__ESBMC_goto_label_2:; // Target\n"
    "i=i + 1; //  // ASSIGN\n"
    "}\n"
    "__ESBMC_goto_label_3:; // Target\n"
    "// dead t;; //  // DEAD\n"
    "} // SCOPE END {2}->{1}\n"
    "}\n"
    "__ESBMC_goto_label_4:; // Target\n"
    "return i; //  // RETURN\n"
    "// ; //  // END_FUNCTION\n");
}

TEST_CASE(
  "goto2c recovers do-while, break, continue and endless loops",
  "[goto2c]")
{
  fixturet f;
  const expr2tc &i = f.i;
  auto num = [&](int v) { return constant_int2tc(f.int_t, BigInt(v)); };
  goto_programt body;
  auto add = [&](goto_program_instruction_typet type) {
    return body.add_instruction(type);
  };
  auto jump = [&](const expr2tc &guard) {
    auto j = add(GOTO);
    j->guard = guard;
    return j;
  };

  auto head = add(ASSIGN);
  head->code = code_assign2tc(i, add2tc(f.int_t, i, num(1)));
  auto brk = jump(equality2tc(i, num(5)));
  auto cont = jump(equality2tc(i, num(6)));
  jump(equality2tc(i, num(7)))->targets.push_back(head);
  auto latch = jump(lessthan2tc(i, num(10)));
  latch->targets.push_back(head);
  auto skip_then = jump(not2tc(greaterthan2tc(i, num(0))));
  add(DECL)->code = code_decl2tc(f.int_t, "c:@F@f@t");
  add(ASSIGN)->code = code_assign2tc(f.t, i);
  add(DEAD)->code = code_dead2tc(f.int_t, "c:@F@f@t");
  auto spin = add(SKIP);
  jump(gen_true_expr())->targets.push_back(spin);
  add(END_FUNCTION);

  brk->targets.push_back(skip_then);
  cont->targets.push_back(latch);
  skip_then->targets.push_back(spin);
  body.update();

  goto2ct goto2c(f.ns, goto_functionst(), true);
  goto2c.preprocess(body);
  CHECK(
    goto2c.translate(body) ==
    "do\n"
    "{\n"
    "__ESBMC_goto_label_1:; // Target\n"
    "i=i + 1; //  // ASSIGN\n"
    "if(i == 5) break;\n"
    "if(i == 6) continue;\n"
    "if(i == 7) goto __ESBMC_goto_label_1; //  // GOTO\n"
    "__ESBMC_goto_label_2:; // Target\n"
    "}\n"
    "while(i < 10);\n"
    "__ESBMC_goto_label_3:; // Target\n"
    "if(i > 0)\n"
    "{\n"
    "{ // SCOPE BEGIN {1}->{2}\n"
    "int t; //  // DECL\n"
    "t=i; //  // ASSIGN\n"
    "// dead t;; //  // DEAD\n"
    "} // SCOPE END {2}->{1}\n"
    "}\n"
    "while(1)\n"
    "{\n"
    "__ESBMC_goto_label_4:; // Target\n"
    "// ; //  // SKIP\n"
    "}\n"
    "// ; //  // END_FUNCTION\n");
}

TEST_CASE("goto2c keeps scope braces balanced", "[goto2c]")
{
  fixturet f;
  f.variable("c:@F@f@a", "a");
  const expr2tc u = f.variable("c:@F@f@u", "u");
  goto_programt body;
  auto add = [&](goto_program_instruction_typet type) {
    return body.add_instruction(type);
  };
  auto decl = [&](const char *id) {
    add(DECL)->code = code_decl2tc(f.int_t, id);
  };
  auto dead = [&](const char *id) {
    add(DEAD)->code = code_dead2tc(f.int_t, id);
  };

  // { int a; } { int t; }: two sibling scopes.
  decl("c:@F@f@a");
  dead("c:@F@f@a");
  decl("c:@F@f@t");
  dead("c:@F@f@t");
  // { int u; if (u) { <u dies here> i = 1; } }
  decl("c:@F@f@u");
  auto test = add(GOTO);
  test->guard = equality2tc(u, gen_zero(f.int_t));
  dead("c:@F@f@u");
  add(ASSIGN)->code = code_assign2tc(f.i, gen_one(f.int_t));
  auto ret = add(RETURN);
  ret->code = code_return2tc(f.i);
  add(END_FUNCTION);
  test->targets.push_back(ret);
  body.update();

  goto2ct goto2c(f.ns, goto_functionst(), true);
  goto2c.preprocess(body);
  // Braces for the if would cut u's scope, so it stays a goto.
  CHECK(
    goto2c.translate(body) ==
    "{ // SCOPE BEGIN {1}->{4}\n"
    "int a; //  // DECL\n"
    "// dead a;; //  // DEAD\n"
    "} // SCOPE END {4}->{1}\n"
    "{ // SCOPE BEGIN {1}->{3}\n"
    "int t; //  // DECL\n"
    "// dead t;; //  // DEAD\n"
    "} // SCOPE END {3}->{1}\n"
    "{ // SCOPE BEGIN {1}->{2}\n"
    "int u; //  // DECL\n"
    "if(u == 0) goto __ESBMC_goto_label_1; //  // GOTO\n"
    "// dead u;; //  // DEAD\n"
    "} // SCOPE END {2}->{1}\n"
    "i=1; //  // ASSIGN\n"
    "__ESBMC_goto_label_1:; // Target\n"
    "return i; //  // RETURN\n"
    "// ; //  // END_FUNCTION\n");
}

TEST_CASE("goto2c leaves a block that would cut a scope as gotos", "[goto2c]")
{
  // do { i = i + 1; int t = i; } while (t != 0), with t dead after the loop.
  fixturet f;
  goto_programt body;
  auto head = body.add_instruction(ASSIGN);
  head->code = code_assign2tc(f.i, add2tc(f.int_t, f.i, gen_one(f.int_t)));
  body.add_instruction(DECL)->code = code_decl2tc(f.int_t, "c:@F@f@t");
  body.add_instruction(ASSIGN)->code = code_assign2tc(f.t, f.i);
  auto latch = body.add_instruction(GOTO);
  latch->guard = notequal2tc(f.t, gen_zero(f.int_t));
  latch->targets.push_back(head);
  body.add_instruction(DEAD)->code = code_dead2tc(f.int_t, "c:@F@f@t");
  body.add_instruction(END_FUNCTION);
  body.update();

  goto2ct goto2c(f.ns, goto_functionst(), true);
  goto2c.preprocess(body);
  CHECK(
    goto2c.translate(body) ==
    "__ESBMC_goto_label_1:; // Target\n"
    "i=i + 1; //  // ASSIGN\n"
    "{ // SCOPE BEGIN {1}->{2}\n"
    "int t; //  // DECL\n"
    "t=i; //  // ASSIGN\n"
    "if(t != 0) goto __ESBMC_goto_label_1; //  // GOTO\n"
    "// dead t;; //  // DEAD\n"
    "} // SCOPE END {2}->{1}\n"
    "// ; //  // END_FUNCTION\n");
}

TEST_CASE("structure recovery bounds its nesting depth", "[goto2c]")
{
  fixturet f;
  goto_programt body;
  std::vector<goto_programt::targett> jumps;
  for (int k = 0; k < 50000; ++k)
  {
    jumps.push_back(body.add_instruction(GOTO));
    jumps.back()->guard = equality2tc(f.i, constant_int2tc(f.int_t, BigInt(k)));
  }
  auto fail = body.add_instruction(SKIP);
  body.add_instruction(END_FUNCTION);
  for (auto &jump : jumps)
    jump->targets.push_back(fail);
  body.update();

  const std::vector<structured_stmtt> tree = recover_structure(body);
  std::size_t depth = 0;
  for (const auto *level = &tree; !level->empty(); level = &level->front().body)
    ++depth;
  CHECK(depth == 257);
}
