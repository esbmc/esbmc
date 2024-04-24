#include <nlohmann/json.hpp>
#include <unordered_set>

namespace SolidityTemplate
{
const std::string parentheis = "()";

/// special variables
// msg
std::string msg_bs = "msg";
std::map<std::string, std::string> msg_mem = {
  {"data", "bytes"},
  {"sender", "address"},
  {"sig", "bytes4"},
  {"value", "uint"}};

// tx
std::string tx_bs = "tx";
std::map<std::string, std::string> tx_mem = {
  {"gasprice", "uint"},
  {"origin", "address"}};

// block
std::string block_bs = "block";
std::map<std::string, std::string> block_mem = {
  {"basefee", "uint"},
  {"chainid", "uint"},
  {"coinbase", "address"},
  {"difficulty", "uint"},
  {"gaslimit", "uint"},
  {"number", "uint"},
  {"prevrandao", "uint"},
  {"timestamp", "uint"}};

/// sepcial functions

std::map<std::string, std::string> block_hash = {{"blockhash", "bytes32"}};
std::map<std::string, std::string> gasleft = {{"gasleft", "uint"}};

// abi
std::string abi_bs = "abi";
std::map<std::string, std::string> ai_mem = {
  {"encode", "bytes"},
  {"encodePacked", "bytes"},
  {"encodeWithSelector", "bytes"},
  {"encodeWithSignature", "bytes"},
  {"encodeCall", "bytes"}}; // {"decode","tuple"},

// byte
std::string byte_bs = "byte";
std::map<std::string, std::string> byte_mem = {
  {"concat", "bytes"},
};

// string
std::string string_bs = "string";
std::map<std::string, std::string> string_mem = {{"concat", "string"}};

// addmod

std::map<std::string, std::string> addmod = {{"addmod", "uint"}};

/// function body
// Always set "stateVariable": true

/* 
    addmod(uint x, uint y, uint k) returns (uint)
    function addmod(uint x, uint y, uint k) public returns (uint)
    {
        assert(k != 0);
        return (x + y) % k;
    }
*/
const std::string addmod_body =
  R"({
                        "body": {
                        "id": 25,
                        "nodeType": "Block",
                        "src": "107:59:0",
                        "statements": [
                            {
                                "expression": {
                                    "arguments": [
                                        {
                                            "commonType": {
                                                "typeIdentifier": "t_uint256",
                                                "typeString": "uint256"
                                            },
                                            "id": 15,
                                            "isConstant": false,
                                            "isLValue": false,
                                            "isPure": false,
                                            "lValueRequested": false,
                                            "leftExpression": {
                                                "id": 13,
                                                "name": "k",
                                                "nodeType": "Identifier",
                                                "overloadedDeclarations": [],
                                                "referencedDeclaration": 7,
                                                "src": "124:1:0",
                                                "typeDescriptions": {
                                                    "typeIdentifier": "t_uint256",
                                                    "typeString": "uint256"
                                                }
                                            },
                                            "nodeType": "BinaryOperation",
                                            "operator": "!=",
                                            "rightExpression": {
                                                "hexValue": "30",
                                                "id": 14,
                                                "isConstant": false,
                                                "isLValue": false,
                                                "isPure": true,
                                                "kind": "number",
                                                "lValueRequested": false,
                                                "nodeType": "Literal",
                                                "src": "129:1:0",
                                                "typeDescriptions": {
                                                    "typeIdentifier": "t_rational_0_by_1",
                                                    "typeString": "int_const 0"
                                                },
                                                "value": "0"
                                            },
                                            "src": "124:6:0",
                                            "typeDescriptions": {
                                                "typeIdentifier": "t_bool",
                                                "typeString": "bool"
                                            }
                                        }
                                    ],
                                    "expression": {
                                        "argumentTypes": [
                                            {
                                                "typeIdentifier": "t_bool",
                                                "typeString": "bool"
                                            }
                                        ],
                                        "id": 12,
                                        "name": "assert",
                                        "nodeType": "Identifier",
                                        "overloadedDeclarations": [],
                                        "referencedDeclaration": -3,
                                        "src": "117:6:0",
                                        "typeDescriptions": {
                                            "typeIdentifier": "t_function_assert_pure$_t_bool_$returns$__$",
                                            "typeString": "function (bool) pure"
                                        }
                                    },
                                    "id": 16,
                                    "isConstant": false,
                                    "isLValue": false,
                                    "isPure": false,
                                    "kind": "functionCall",
                                    "lValueRequested": false,
                                    "nameLocations": [],
                                    "names": [],
                                    "nodeType": "FunctionCall",
                                    "src": "117:14:0",
                                    "tryCall": false,
                                    "typeDescriptions": {
                                        "typeIdentifier": "t_tuple__$",                     
                                        "typeString": "tuple)" +
  parentheis + R"("
                                    }
                                },
                                "id": 17,
                                "nodeType": "ExpressionStatement",
                                "src": "117:14:0"
                            },
                            {
                                "expression": {
                                    "commonType": {
                                        "typeIdentifier": "t_uint256",
                                        "typeString": "uint256"
                                    },
                                    "id": 23,
                                    "isConstant": false,
                                    "isLValue": false,
                                    "isPure": false,
                                    "lValueRequested": false,
                                    "leftExpression": {
                                        "components": [
                                            {
                                                "commonType": {
                                                    "typeIdentifier": "t_uint256",
                                                    "typeString": "uint256"
                                                },
                                                "id": 20,
                                                "isConstant": false,
                                                "isLValue": false,
                                                "isPure": false,
                                                "lValueRequested": false,
                                                "leftExpression": {
                                                    "id": 18,
                                                    "name": "x",
                                                    "nodeType": "Identifier",
                                                    "overloadedDeclarations": [],
                                                    "referencedDeclaration": 3,
                                                    "src": "149:1:0",
                                                    "typeDescriptions": {
                                                        "typeIdentifier": "t_uint256",
                                                        "typeString": "uint256"
                                                    }
                                                },
                                                "nodeType": "BinaryOperation",
                                                "operator": "+",
                                                "rightExpression": {
                                                    "id": 19,
                                                    "name": "y",
                                                    "nodeType": "Identifier",
                                                    "overloadedDeclarations": [],
                                                    "referencedDeclaration": 5,
                                                    "src": "153:1:0",
                                                    "typeDescriptions": {
                                                        "typeIdentifier": "t_uint256",
                                                        "typeString": "uint256"
                                                    }
                                                },
                                                "src": "149:5:0",
                                                "typeDescriptions": {
                                                    "typeIdentifier": "t_uint256",
                                                    "typeString": "uint256"
                                                }
                                            }
                                        ],
                                        "id": 21,
                                        "isConstant": false,
                                        "isInlineArray": false,
                                        "isLValue": false,
                                        "isPure": false,
                                        "lValueRequested": false,
                                        "nodeType": "TupleExpression",
                                        "src": "148:7:0",
                                        "typeDescriptions": {
                                            "typeIdentifier": "t_uint256",
                                            "typeString": "uint256"
                                        }
                                    },
                                    "nodeType": "BinaryOperation",
                                    "operator": "%",
                                    "rightExpression": {
                                        "id": 22,
                                        "name": "k",
                                        "nodeType": "Identifier",
                                        "overloadedDeclarations": [],
                                        "referencedDeclaration": 7,
                                        "src": "158:1:0",
                                        "typeDescriptions": {
                                            "typeIdentifier": "t_uint256",
                                            "typeString": "uint256"
                                        }
                                    },
                                    "src": "148:11:0",
                                    "typeDescriptions": {
                                        "typeIdentifier": "t_uint256",
                                        "typeString": "uint256"
                                    }
                                },
                                "functionReturnParameters": 11,
                                "id": 24,
                                "nodeType": "Return",
                                "src": "141:18:0"
                            }
                        ]
                    },
                    "functionSelector": "9796df37",
                    "id": 26,
                    "implemented": true,
                    "kind": "function",
                    "modifiers": [],
                    "name": "addmod",
                    "nameLocation": "54:6:0",
                    "nodeType": "FunctionDefinition",
                    "parameters": {
                        "id": 8,
                        "nodeType": "ParameterList",
                        "parameters": [
                            {
                                "constant": false,
                                "id": 3,
                                "mutability": "mutable",
                                "name": "x",
                                "nameLocation": "66:1:0",
                                "nodeType": "VariableDeclaration",
                                "src": "61:6:0",
                                "stateVariable": true,
                                "storageLocation": "default",
                                "typeDescriptions": {
                                    "typeIdentifier": "t_uint256",
                                    "typeString": "uint256"
                                },
                                "typeName": {
                                    "id": 2,
                                    "name": "uint",
                                    "nodeType": "ElementaryTypeName",
                                    "src": "61:4:0",
                                    "typeDescriptions": {
                                        "typeIdentifier": "t_uint256",
                                        "typeString": "uint256"
                                    }
                                },
                                "visibility": "internal"
                            },
                            {
                                "constant": false,
                                "id": 5,
                                "mutability": "mutable",
                                "name": "y",
                                "nameLocation": "74:1:0",
                                "nodeType": "VariableDeclaration",
                                "src": "69:6:0",
                                "stateVariable": true,
                                "storageLocation": "default",
                                "typeDescriptions": {
                                    "typeIdentifier": "t_uint256",
                                    "typeString": "uint256"
                                },
                                "typeName": {
                                    "id": 4,
                                    "name": "uint",
                                    "nodeType": "ElementaryTypeName",
                                    "src": "69:4:0",
                                    "typeDescriptions": {
                                        "typeIdentifier": "t_uint256",
                                        "typeString": "uint256"
                                    }
                                },
                                "visibility": "internal"
                            },
                            {
                                "constant": false,
                                "id": 7,
                                "mutability": "mutable",
                                "name": "k",
                                "nameLocation": "82:1:0",
                                "nodeType": "VariableDeclaration",
                                "src": "77:6:0",
                                "stateVariable": true,
                                "storageLocation": "default",
                                "typeDescriptions": {
                                    "typeIdentifier": "t_uint256",
                                    "typeString": "uint256"
                                },
                                "typeName": {
                                    "id": 6,
                                    "name": "uint",
                                    "nodeType": "ElementaryTypeName",
                                    "src": "77:4:0",
                                    "typeDescriptions": {
                                        "typeIdentifier": "t_uint256",
                                        "typeString": "uint256"
                                    }
                                },
                                "visibility": "internal"
                            }
                        ],
                        "src": "60:24:0"
                    },
                    "src": "45:121:0",
                    "stateMutability": "nonpayable",
                    "virtual": false,
                    "visibility": "public"
    })";

}; // namespace SolidityTemplate
