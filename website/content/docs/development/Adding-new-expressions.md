---
title: Adding New Expressions
---

Adding expressions can be very confusing as there many places that need to be updated:

- Frontends need to use the expression.
- irep2 should be adapted to hold the expressions
- irep -> irep2 migrations needs to be implemented if you want to implement it directly into your language
- simplifications need to be implemented
- irep2 -> smt needs to be implemented

The tricky part is that you can only test the expression once the full stack is complete.

So I will be summarizing my steps adding support to quantifiers in ESBMC, including how the development process.


#  Quantifiers

I will introduce two expressions: `forall(symbol,bool-expr)` and `exists(symbol,bool-expr)`.

## Frontend

First, I will be exposing these functions to the C frontend. This will be done through two intrinsic functions:

```c
_Bool __ESBMC_forall(void*, _Bool);
_Bool __ESBMC_exists(void*, _Bool);
```

This is just a matter of adding the declaration at `clang_c_language.cpp`

Note: if the language that you are going to use has support for these operators then you just convert them directly. No need to create an intrinsic.

In `builtin_functions.cpp` we can hijack these functions and convert them into the expression. That file, `src/goto-programs/builtin_functions.cpp`, does static replacement at the goto level; "dynamic" replacement happens in symex, under `src/goto-symex/engine/builtin_functions/`. For quantifiers, I want to replace them at the goto level.

Of course... we can't generate anything yet. I suggest making them return nondet for now. 

```cpp
// Quantifiers
  if (base_name == "__ESBMC_forall" || base_name == "__ESBMC_exists")
  {
    // make it a side effect if there is an LHS
    if (lhs.is_nil())
      return;

    exprt rhs = side_effect_expr_nondett(lhs.type());
    rhs.location() = function.location();

    code_assignt assignment(lhs, rhs);
    assignment.location() = function.location();
    copy(assignment, ASSIGN, dest);
    return;
  }
```

## Irep1

Next step is to create the expression at irep1 level. This is very simple as irep1 is pretty much a JSON

```cpp
    exprt rhs = exprt("forall", typet("bool"));
    rhs.copy_to_operands(arguments[0]);
    rhs.copy_to_operands(arguments[1]);
```

That's all. Running a program now should result in issues at migrate.

## Irep2

Before solving the migration issue, we need to create the expression at irep2. Irep2 relies on lots of metaprogramming to generate the expressions in a sane way. A custom LSP for esbmc project might be great to sort this out! Adding new expressions/types is not a well documented process and the sideffects are not well known.

The current checklist is "Adding a new node" in `src/irep2/README.md`. For
`forall` it has three steps.

First, append a row for the expression to `src/irep2/expr_kinds.inc`. This file
is the single list of expression kinds; the `forall_id` entry of the
`expr2t::expr_ids` enum and the `is_forall2t` and `to_forall2t` helpers are
generated from it:

```cpp
IREP2_EXPR(forall,                "forall")
```

Second, define the class in `src/irep2/irep2_expr.h`, together with an
`irep_typedefs(forall);` line that generates the `forall2tc(...)` constructor
helper. The class inherits from `expr2t`, declares its fields, and lists them in
a `fields` tuple in the same order as the constructor parameters:

```cpp
class forall2t : public expr2t
{
public:
  expr2tc side_1, side_2;
  forall2t(const type2tc &type, const expr2tc &sym, const expr2tc &predicate)
    : expr2t(type, forall_id), side_1(sym), side_2(predicate)
  {
  }
  forall2t(const forall2t &ref) = default;

  static constexpr auto fields =
    std::make_tuple(&expr2t::type, &forall2t::side_1, &forall2t::side_2);
  static std::string field_names[esbmct::num_type_fields];
};
```

Third, add the `field_names` table in `src/irep2/irep2_expr.cpp`, one string
per field after `type`, in tuple order, padded with empty strings:
`std::string forall2t::field_names[esbmct::num_type_fields] = {"symbol", "predicate", "", "", ""};`.

This is the barebones to have a code that compiles. We can now update the migration to use this new expression.

```cpp
else if (expr.id() == "forall")
  {
    type = migrate_type(expr.type());
    expr2tc args[2];
    migrate_expr(expr.op0(), args[0]);
    migrate_expr(expr.op1(), args[1]);
    new_expr_ref = forall2tc(type, args[0], args[1]);
  }
```

Also, you need to irep2 -> irep1 convertion

```cpp
case expr2t::forall_id:
  {
    const forall2t &ref2 = to_forall2t(ref);
    exprt back("forall", migrate_type_back(ref2.type));
    back.copy_to_operands(migrate_expr_back(ref2.side_1));
    back.copy_to_operands(migrate_expr_back(ref2.side_2));
    return back;
  }
```

This will generate GOTO programs with

```
        ASSIGN return_value$___ESBMC_forall$1=forall
  * type: bool
  * operands: 
    0: typecast
        * type: pointer
            * subtype: empty
        * operands: 
          0: address_of
              * type: pointer
                  * subtype: signedbv
                      * width: 32
              * operands: 
                0: symbol
                    * type: signedbv
                        * width: 32
                    * identifier: c:exists.c@15@F@main@x
        * rounding_mode: symbol
            * type: signedbv
                * width: 32
            * identifier: c:@__ESBMC_rounding_mode
    1: typecast
        * type: bool
        * operands: 
          0: >
              * type: bool
              * operands: 
                0: symbol
                    * type: signedbv
                        * width: 32
                    * identifier: c:exists.c@15@F@main@x
                1: constant
                    * type: signedbv
                        * width: 32
                    * value: 00000000000000000000000000000000
        * rounding_mode: symbol
            * type: signedbv
                * width: 32
            * identifier: c:@__ESBMC_rounding_mode;
```

Later we will fix the pretty printing.

## SMT

If we try to run esbmc with:

```c
int main()
{
  int x;
  if (x > 10)
  {
    __ESBMC_assert(__ESBMC_forall(&x, x > 0), "test");
  }
}
```
It will result in a crash. We still need to define the semantics of the forall. We can do it at symex, smt, simplifier... or all of them. For now lets start with smt only.

This is a special case, SMT has theories to support FORALL, which means that we need to extend our smt conversion layer to it. At `smt_conv.h`
we can add:

```cpp
virtual smt_astt mk_forall(smt_astt symbols, smt_astt body);
virtual smt_astt mk_exists(smt_astt symbols, smt_astt body);

smt_astt mk_forall(smt_astt symbols, smt_astt body)
{
  (void)symbols;
  (void)body;
  abort();
}
```

At smt_conv, we can add the switch case for how to convert the expression into smt

```cpp
case expr2t::forall_id:
  {
 const forall2t &f = to_forall2t(expr);
    a = mk_forall(convert_ast(f.side_1), convert_ast(f.side_2));
    break;
  }
```

Then for solver specific... we need to implement using their APIs