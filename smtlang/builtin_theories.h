/*******************************************************************\

Module: SMT-LIB Builtin Theories  

Author: CM Wintersteiger

\*******************************************************************/

#ifndef BUILTIN_THEORIES_H_
#define BUILTIN_THEORIES_H_

#include <expr.h>

class smt_theory {
  public:
    virtual ~smt_theory( void ) {};    
    virtual const char** get_sorts( void ) const = 0;
    virtual const char*** get_predicates( void ) const = 0;
    virtual const char*** get_functions( void ) const = 0;
    
    virtual bool check_predicates(const std::string&, const typet&);
    virtual typet check_functions(const std::string&, const typet&);
    virtual bool check_sorts(const std::string&);
};

class smt_theory_Ints : public smt_theory {
  public:
    static const char* sorts[];
    static const char** predicates[];
    static const char** functions[];
    
    virtual const char** get_sorts( void ) const { return sorts; };
    virtual const char*** get_predicates( void ) const { return predicates; };
    virtual const char*** get_functions( void ) const { return functions; };
};

class smt_theory_Reals : public smt_theory {
  public:
    static const char* sorts[];
    static const char** predicates[];
    static const char** functions[];
    
    virtual const char** get_sorts( void ) const { return sorts; };
    virtual const char*** get_predicates( void ) const { return predicates; };
    virtual const char*** get_functions( void ) const { return functions; };
};

class smt_theory_Int_Arrays : public smt_theory {
  public:
    static const char* sorts[];
    static const char** predicates[];
    static const char** functions[];
    
    virtual const char** get_sorts( void ) const { return sorts; };
    virtual const char*** get_predicates( void ) const { return predicates; };
    virtual const char*** get_functions( void ) const { return functions; };
};

class smt_theory_Int_ArraysEx : 
  public smt_theory_Int_Arrays {}; // this one is the same
  
class smt_theory_Int_Int_Real_Array_ArraysEx: public smt_theory {
  public:
    static const char* sorts[];
    static const char** predicates[];
    static const char** functions[];
    
    virtual const char** get_sorts( void ) const { return sorts; };
    virtual const char*** get_predicates( void ) const { return predicates; };
    virtual const char*** get_functions( void ) const { return functions; };
};

class smt_theory_Fixed_Size_BitVectors32 : public smt_theory {
  public:
    static const char* sorts[];
    static const char** predicates[];
    static const char** functions[];
    
    virtual const char** get_sorts( void ) const { return sorts; };
    virtual const char*** get_predicates( void ) const { return predicates; };
    virtual const char*** get_functions( void ) const { return functions; };
    
    virtual bool check_predicates(const std::string&, const typet&);
    virtual typet check_functions(const std::string&, const typet&);
    virtual bool check_sorts(const std::string&);
    
    static int extract_bv_size( const std::string& );
    static std::pair<int, int> extract_bv_range( const std::string& );
};

#endif /*BUILTIN_THEORIES_H_*/
