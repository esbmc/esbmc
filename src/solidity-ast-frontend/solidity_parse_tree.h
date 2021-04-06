/*******************************************************************\

Module: Class to store declaration in parse_tree and
        convert declarations to symbols

Author: Kunjian Song

\*******************************************************************/

#ifndef SOLIDITY_AST_FRONTEND_SOLIDITY_PARSE_TREE_H
#define SOLIDITY_AST_FRONTEND_SOLIDITY_PARSE_TREE_H

#include <solidity-ast-frontend/solidity_declaration.h>

class solidity_parse_treet
{
public:
    // the declarations
    typedef std::list<solidity_declarationt> declarationst;
    declarationst declarations;

    void swap(solidity_parse_treet &other);
    void clear();

    void output(std::ostream &out) const
    {
        //TODO
        assert(!"come back and continue - solidity_parse_treet::output");
        for(const auto &declaration : declarations)
        {
            solidity_symbolt tmp;
            declaration.to_symbol(tmp);
            //out << tmp;
        }
    }
};

#endif
