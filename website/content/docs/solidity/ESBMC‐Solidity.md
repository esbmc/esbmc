---
title: Solidity
---

- 🟩: Easy to implement
- 🟧: Medium
- 🟥: Hard
- ⚠️: Important, highest priority. Because they are commonly seen in real-world smart contracts.
- ❔: Not sure
- ~~AA~~: Supported

## Frontend Construction

### TODO List

- Type
  - [Rational Literals](https://docs.soliditylang.org/en/v0.8.23/types.html#rational-and-integer-literals) 🟩
  - [User-defined Value Types](https://docs.soliditylang.org/en/v0.8.23/types.html#user-defined-value-types) 🟧
    - Almost identical to alias or `typedef`.
  - [Mapping](https://docs.soliditylang.org/en/v0.8.23/types.html#mapping-types) 🟥⚠️
    - **Can also be valuable for C++/Python frontend**.
  - [Function Types Members](https://docs.soliditylang.org/en/v0.8.23/types.html#function-types)❔
    - `.address` 🟩⚠️
    - `.selector` ❔
- ~~[Events](https://docs.soliditylang.org/en/v0.8.23/abi-spec.html#events)❔⚠️~~
  - I would say it is 🟩, because it does not really affect the verification, only outputting logs. Maybe just parse it and do nothing
- ~~[Errors](https://docs.soliditylang.org/en/v0.8.23/abi-spec.html#errors)🟩⚠️~~
  - ❔Do we need to implement roll back❔
- [Units and Globally Available Variables](https://docs.soliditylang.org/en/v0.8.23/units-and-global-variables.html#units-and-globally-available-variables)
  - [Ether Units](https://docs.soliditylang.org/en/v0.8.23/units-and-global-variables.html#ether-units) 🟧
  - [Time Units](https://docs.soliditylang.org/en/v0.8.23/units-and-global-variables.html#time-units) 🟧
    - Maybe 🟩. I am just not sure what they should be converted to.
  - [Block and Transaction Properties](https://docs.soliditylang.org/en/v0.8.23/units-and-global-variables.html#block-and-transaction-properties) 🟩⚠️
    - Can be regarded as built-in variables which should be preloaded before parsing (?)
    - the value of these built-in properties should be **non-deterministic**. However, might construct an interface to allow user-defined value in the future.
  - ~~[ABI Encoding and Decoding Functions](https://docs.soliditylang.org/en/v0.8.23/units-and-global-variables.html#abi-encoding-and-decoding-functions) 🟧⚠️~~
  - [Members of bytes](https://docs.soliditylang.org/en/v0.8.23/units-and-global-variables.html#members-of-bytes) 🟥
    - `byte.concat`
    - `byte.length`
    - `byte.push`
    - `byte.pop` ⚠️ Very important. There is a type of vulnerability called `popping an empty array`. (**0x31**) 
  - [Members of string](https://docs.soliditylang.org/en/v0.8.23/units-and-global-variables.html#members-of-string) 🟩
    - `string.concat`: convert to `c:@F@strncat`
  - [Mathematical and Cryptographic Functions](https://docs.soliditylang.org/en/v0.8.23/units-and-global-variables.html#mathematical-and-cryptographic-functions) 🟧⚠️
  - [Members of Address Types](https://docs.soliditylang.org/en/v0.8.23/units-and-global-variables.html#members-of-address-types) ❔⚠️
  - [Contract-related](https://docs.soliditylang.org/en/v0.8.23/units-and-global-variables.html#contract-related) ❔⚠️
  - [Type Information](https://docs.soliditylang.org/en/v0.8.23/units-and-global-variables.html#type-information) ❔
- Function
  - [Function Calls with Named Parameters](https://docs.soliditylang.org/en/v0.8.23/control-structures.html#function-calls-with-named-parameters) 🟩
  - [Destructuring Assignments and Returning Multiple Values](https://docs.soliditylang.org/en/v0.8.23/control-structures.html#destructuring-assignments-and-returning-multiple-values) 🟥⚠️
    - Basically, tuple in python
    - **Can also be valuable for C++/Python frontend**.
  - [Omitted Names in Function Definitions](https://docs.soliditylang.org/en/v0.8.23/control-structures.html#omitted-names-in-function-definitions) 🟥
  - [Scoping and Declarations](https://docs.soliditylang.org/en/v0.8.23/control-structures.html#scoping-and-declarations) 🟩⚠️
  - [Call a zero-initialized variable of internal function type](https://ethereum.stackexchange.com/questions/47009/call-a-zero-initialized-variable-of-internal-function-type) ⚠️ 
    - Another vulnerability type (**0x51**)
- [Function Modifiers](https://docs.soliditylang.org/en/v0.8.23/structure-of-a-contract.html#function-modifiers) 🟧
- Interface ❔⚠️
- Abstract ❔⚠️
- Keywords
  - delete
  - super
  - this
  - ...

## Known Bugs
- ~~Inheritance. Completely broken.~~
  - The override and virtual still contain bugs.
- Incomplete message output related to struct. (e.g. Assertion `struct.id == 1` failed is reported as `struct. == 1`)
- [out-of-bounds Bytes](https://docs.soliditylang.org/en/v0.8.23/control-structures.html#panic-via-assert-and-error-via-require) (**0x32**) 🟧⚠️
- (need investigation) We did not implement the rollback features in Solidity. Will it affect the verification result?

## Resource List
- [cprover: Background Concepts](https://diffblue.github.io/cbmc//background-concepts.html)
- [ESBMC-solidity: an SMT-based model checker for solidity smart contracts](https://ssvlab.github.io/lucasccordeiro/papers/icse2022.pdf)
- [ESBMC Document](http://esbmc.org)
- [solidity/docs at develop · ethereum/solidity](https://github.com/ethereum/solidity/tree/develop/docs)

### Solidity Error Code
```
0x01: If you call assert with an argument that evaluates to false.
0x11: If an arithmetic operation results in underflow or overflow outside of an unchecked { ... } block.
0x12: If you divide or modulo by zero (e.g. 5 / 0 or 23 % 0).
0x21: If you convert a value that is too big or negative into an enum type.
0x31: If you call .pop() on an empty array.
0x32: If you access an array, bytesN or an array slice at an out-of-bounds or negative index (i.e. x[i] where i >= x.length or i < 0).
0x41: If you allocate too much memory or create an array that is too large.
0x51: If you call a zero-initialized variable of internal function type
```
