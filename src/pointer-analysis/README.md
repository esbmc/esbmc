Here, we describe some informal notes about the ESBMC's memory model.

The C specification demands that absolutely every `data object` has an equivalent representation as a sequence of bytes, whether it is a struct, union, or float. This is how `memcpy` works: you pass it two pointers, and it copies them byte by byte. What this also means is that you can:
 
* Take a pointer to an arbitrary data object
* Cast it to a "char" pointer
* Add an offset to the char pointer so that it points into the data
 object
* Dereference it with predictable results.

The trouble comes from the fact that SMT solvers do not like this one bit. It would be OK to treat all variable storage as one big SMT array of bytes and repeatedly store and load from it. Arbitrary byte accesses would be regular array access to memory. However, we'd almost certainly need better solving performance (the SMT solver would have to explore state space in a fixed order).

What ESBMC does instead is use the primitives that SMT provides (bitvectors, arrays, possibly floatbvs) as variables to store values. That means that whenever we have code like this:

```
 int foo, bar;
 char *baz = (nondet_bool()) ? (char*)&foo : (char *)&bar;
 unsigned int idx = nondet_uint();
 __ESBMC_assume(idx < sizeof(int));
 baz += idx;
 *baz = 1;
```

We must translate the `store` through baz into an arbitrary byte access to an arbitrary SMT primitive as either a load or store. Assuming a store, there are at least three types in play:
* The type that the `rhs` of the assignment evaluates to,
* The type of the pointer that is being dereferenced in the store,
* The type of the primitive backing the data object.


Taking a look at `dereference.cpp` then: for the unary operand of the `*` operator, this digs down through each symbol that it might evaluate to, just in case it is:

```
 *(nondet_bool() ? foo : bar)
```

It produces an appropriate guard along the way. If the type of the dereference is not a scalar (i.e., it is an array or struct), it goes via `dereference_expr_nonscalar`, which collects a list of `index2t`'s or `member2t`'s applied to the base expression. Any expression that has an array or struct type might be a plain symbol or a field in some struct, for example:

```
 struct xyzzy {
   int foo;
   int bar[4];
 };
 struct qux {
   struct xyzzy quux[2];
 };
 qux *croix;
 (void)croix->quux[1].bar;
```

Here `dereference_expr_nonscalar` would pick out that 'croix' is being dereferenced, but with a series of indexes and members that identified the field being accessed.

This all feeds into `dereferencet::dereference`, which takes:

* The base expression that is being dereferenced; usually a symbol,
* The type of the expression being dereferenced, i.e., the type that
  we're going to need to pick out of the underlying primitives,
* The control-flow guard for this `deref`,
* The mode, a read, write, "free" or "give me a list of objects" mode.
* The offset applied to the address being dereferenced. i.e., if dereferencing a struct field, the offset into the struct.

A list of all variable names that the base expression can point to is fetched. These variable names are `l1` renamed variables: they identify what the C spec might term `storage`, an `lvalue`, or data object. The points-at information comes from `the value_sett` tracking. Each pointed-at object is passed to `build_reference_to`; then they are all chained together with a big `if-then-else`. The resulting expression describes the data being accessed in terms of SMT primitives.

All of that is easy, though: the hard stuff is in `dereferencet::build_reference_to`. There we have the core problem of:

* Here is an SMT primitive (`what`),
* Here is the offset into it we want to read/write (`lexical_offset`)
* Here is the type of data to read/write (`type`)

We call `build_reference_rec` with that information. Look at the switch statement; here, we have the cross-product of all the types and different configurations of information encapsulated in one memory load.

In the switch-statement, some facts are accumulated about the types we are dealing with into a single flags number, which is switched on. We have to consider:

* The type of the underlying primitive (`flag_src_*`)
* The type that the dereference needs to evaluate to (`flag_dst_*`), i.e., what data is actually being loaded/stored,
* Whether the offset into the primitive is a constant or not. A constant would be `foo->bar[4]`, non-constant would be `foo->bar[nondet_int()]`.

Take the first one:

```
flag_src_scalar | flag_dst_scalar | flag_is_const_offs
```

The underlying primitive is a scalar (bitvector, float, bool). The memory access wants a scalar, and we know the offset into the underlying object is constant. That calls a function designed to deal with that.

Consider the next one:

```
 flag_src_struct | flag_dst_scalar | flag_is_const_offs:
```

We are extracting a scalar value from a structure, at a known offset. That corresponds to accessing a known field in a struct, such as:

```
croix->quux[1].bar
```

However, it might just as quickly have been a plain integer dereference:

```
 *i
```

That just so happened to point deep into a structure.

The cases in `build_reference_rec` become increasingly hellish as you go down: for example, `C99` quite happily allows you to make assignments with struct type, and you can point a pointer (almost) anywhere in memory and assign that type in and out. For example:

```
 char *foo = malloc(16);
 struct bar {
   int a, b, c, d;
 };
 struct baz = { 1, 2, 3, 4 };
 *((struct bar *)foo) = baz;
```

Involves decomposing a structure SMT primitive into sixteen individual bytes and then assigning them to the byte array returned by malloc. A lot of the more complicated cases boil down to `dereferencet::stitch_together_from_byte_array`, where the underlying primitive has an expression produced for each byte of it, which is then
turned into one massive expression identifying the set of bytes that might be accessed, for each offset.

