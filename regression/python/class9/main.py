class BankAccount:
    def __init__(self, owner: str):
        self.owner = owner
        
account = BankAccount("Alice")
assert account.owner == "Alice"

#strcmp call:

# sideeffect
#   * type: signedbv
#       * width: 32
#   * operands: 
#     0: symbol
#         * type: code
#             * arguments: 
#               0: argument
#                   * type: pointer
#                       * subtype: signedbv
#                           * width: 8
#                           * #constant: 1
#                           * #cpp_type: signed_char
#                   * #location: 
#                     * file: /home/bruno/projects/esbmc/src/c2goto/library/string.c
#                     * line: 94
#                     * function: strcmp
#                     * column: 12
#                   * #base_name: p1
#                   * #identifier: c:string.c@1408@F@strcmp@p1
#               1: argument
#                   * type: pointer
#                       * subtype: signedbv
#                           * width: 8
#                           * #constant: 1
#                           * #cpp_type: signed_char
#                   * #location: 
#                     * file: /home/bruno/projects/esbmc/src/c2goto/library/string.c
#                     * line: 94
#                     * function: strcmp
#                     * column: 28
#                   * #base_name: p2
#                   * #identifier: c:string.c@1424@F@strcmp@p2
#             * return_type: signedbv
#                 * width: 32
#                 * #cpp_type: signed_int
#         * name: strcmp
#         * identifier: c:@F@strcmp
#     1: arguments
#         * operands: 
#           0: member
#               * type: pointer
#                   * subtype: signedbv
#                       * width: 8
#               * operands: 
#                 0: symbol
#                     * type: symbol
#                         * identifier: tag-BankAccount
#                     * identifier: py:main.py@account
#               * component_name: owner
#           1: address_of
#               * type: pointer
#                   * subtype: signedbv
#                       * width: 8
#               * operands: 
#                 0: index
#                     * type: signedbv
#                         * width: 8
#                     * operands: 
#                       0: constant
#                           * type: array
#                               * size: constant
#                                   * type: unsignedbv
#                                       * width: 64
#                                   * value: 0000000000000000000000000000000000000000000000000000000000000101
#                                   * #cformat: 5
#                               * subtype: signedbv
#                                   * width: 8
#                           * operands: 
#                             0: constant
#                                 * type: signedbv
#                                     * width: 8
#                                 * value: 01000001
#                                 * #cformat: 65
#                             1: constant
#                                 * type: signedbv
#                                     * width: 8
#                                 * value: 01101100
#                                 * #cformat: 108
#                             2: constant
#                                 * type: signedbv
#                                     * width: 8
#                                 * value: 01101001
#                                 * #cformat: 105
#                             3: constant
#                                 * type: signedbv
#                                     * width: 8
#                                 * value: 01100011
#                                 * #cformat: 99
#                             4: constant
#                                 * type: signedbv
#                                     * width: 8
#                                 * value: 01100101
#                                 * #cformat: 101
#                       1: constant
#                           * type: signedbv
#                               * width: 64
#                           * value: 0000000000000000000000000000000000000000000000000000000000000000
#   * statement: function_call
