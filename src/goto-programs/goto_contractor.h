#ifndef ESBMC_GOTO_CONTRACTOR_H
#define ESBMC_GOTO_CONTRACTOR_H

#include <iostream>
#include <util/goto_expr_factory.h>
#include <goto-programs/goto_functions.h>
#include <util/algorithms.h>
#include <util/message.h>
#include <goto-programs/goto_loops.h>
#include <goto-programs/remove_no_op.h>
#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_functions.h>
#include <goto-programs/loopst.h>
#include <util/std_types.h>
#include <ibex.h>
#include <ibex/ibex_Interval.h>
#include <ibex/ibex_Expr.h>
#include <ibex/ibex_Ctc.h>
#include <irep2/irep2.h>
#include <util/type_byte_size.h>
#include <goto-programs/abstract-interpretation/interval_domain.h>
#include <goto-programs/abstract-interpretation/interval_analysis.h>
#include <limits>

void goto_contractor(
  goto_functionst &goto_functions,
  const namespacet &namespacet,
  const optionst &options);

class vart
{
private:
  ibex::Interval interval;
  std::string var_name;
  expr2tc symbol;
  size_t index;
  bool interval_changed;

public:
  size_t getIndex() const;

  vart();

public:
  vart(const string &varName, const expr2tc &symbol, const size_t &index);
  const ibex::Interval &getInterval() const;
  void setInterval(const ibex::Interval &interval);
  bool isIntervalChanged() const;
  void setIntervalChanged(bool intervalChanged);
  const expr2tc &getSymbol() const;
  void dump();
};

class Contractor
{
  ibex::Ctc *outer;
  ibex::Ctc *inner;
  unsigned int location;

public:
  Contractor()
  {
    outer = nullptr;
    location = 0;
    inner = nullptr;
  }
  Contractor(ibex::Ctc *c, unsigned int loc)
  {
    outer = c;
    location = loc;
    inner = get_complement_contractor(c);
  }
  Contractor(ibex::Ctc *c)
  {
    outer = c;
    location = 0;
    inner = nullptr;
  }
  ~Contractor()
  {
    //clean up
    for (auto it = vector_ctc.rbegin(); it != vector_ctc.rend(); ++it)
      delete (*it);
    for (auto it = vector_nc.rbegin(); it != vector_nc.rend(); ++it)
      delete (*it);
    for (auto it = vector_f.rbegin(); it != vector_f.rend(); ++it)
      delete (*it);
  }

  void set_outer(ibex::Ctc *outer)
  {
    Contractor::outer = outer;
  }
  void set_inner(ibex::Ctc *inner)
  {
    Contractor::inner = inner;
  }
  void set_location(unsigned int location)
  {
    Contractor::location = location;
  }
  ibex::Ctc *get_outer() const
  {
    return outer;
  }
  ibex::Ctc *get_inner() const
  {
    return inner;
  }
  unsigned int get_location() const
  {
    return location;
  }

private:
  //Cleanup
  std::vector<ibex::Function *> vector_f;
  std::vector<ibex::NumConstraint *> vector_nc;
  std::vector<ibex::Ctc *> vector_ctc;

  ibex::Ctc *get_complement_contractor(ibex::Ctc *c)
  {
    if (auto ctc_compo = dynamic_cast<ibex::CtcCompo *>(c))
    {
      ibex::Array<ibex::Ctc> list_of_contractors;
      for (auto &it : ctc_compo->list)
        list_of_contractors.add(*get_complement_contractor(&it));
      auto ctc_union = new ibex::CtcUnion(list_of_contractors);
      vector_ctc.push_back(ctc_union);
      return c;
    }
    else if (auto ctc_union = dynamic_cast<ibex::CtcUnion *>(c))
    {
      ibex::Array<ibex::Ctc> list_of_contractors;
      for (auto &it : ctc_union->list)
        list_of_contractors.add(*get_complement_contractor(&it));
      auto ctc_compo = new ibex::CtcCompo(list_of_contractors);
      vector_ctc.push_back(ctc_compo);
      return ctc_compo;
    }
    else if (auto fwdbwd = dynamic_cast<ibex::CtcFwdBwd *>(c))
    {
      ibex::NumConstraint *ctr;
      ibex::CtcFwdBwd *contractor;
      switch (fwdbwd->ctr.op)
      {
      case ibex::GEQ:
        ctr = new ibex::NumConstraint(fwdbwd->ctr.f, ibex::LT);
        vector_nc.push_back(ctr);
        contractor = new ibex::CtcFwdBwd(*ctr);
        vector_ctc.push_back(contractor);
        return contractor;
      case ibex::GT:
        ctr = new ibex::NumConstraint(fwdbwd->ctr.f, ibex::LEQ);
        vector_nc.push_back(ctr);
        contractor = new ibex::CtcFwdBwd(*ctr);
        vector_ctc.push_back(contractor);
        return contractor;
      case ibex::LEQ:
        ctr = new ibex::NumConstraint(fwdbwd->ctr.f, ibex::GT);
        vector_nc.push_back(ctr);
        contractor = new ibex::CtcFwdBwd(*ctr);
        vector_ctc.push_back(contractor);
        return contractor;
      case ibex::LT:
        ctr = new ibex::NumConstraint(fwdbwd->ctr.f, ibex::GEQ);
        vector_nc.push_back(ctr);
        contractor = new ibex::CtcFwdBwd(*ctr);
        vector_ctc.push_back(contractor);
        return contractor;
      case ibex::EQ:
        ctr = new ibex::NumConstraint(fwdbwd->ctr.f, ibex::GT);
        auto ctr2 = new ibex::NumConstraint(fwdbwd->ctr.f, ibex::LT);
        auto *side1 = new ibex::CtcFwdBwd(*ctr);
        auto *side2 = new ibex::CtcFwdBwd(*ctr2);
        auto ctc_union = new ibex::CtcUnion(*side1, *side2);
        //for clean up
        vector_nc.push_back(ctr);
        vector_nc.push_back(ctr2);
        vector_ctc.push_back(side1);
        vector_ctc.push_back(side2);
        vector_ctc.push_back(ctc_union);

        return ctc_union;
      }
    }
    else
    {
      log_debug("contractor", "Contractors: Unsupported Ctc type");
    }
    return nullptr;
  }
};

class Contractors
{
private:
  Contractor *c;
  std::list<Contractor *> contractors;

public:
  Contractors()
  {
    c = nullptr;
  }
  ~Contractors()
  {
    if (c != nullptr)
    {
      if (c->get_outer() != nullptr)
        delete (c->get_outer());
      if (c->get_inner() != nullptr)
        delete (c->get_inner());
      delete (c);
    }
    for (auto cont : contractors)
      delete (cont);
  }
  Contractor *get_contractors_up_to_loc(unsigned int loc)
  {
    c = new Contractor();

    auto size = contractors.size();
    ibex::Array<ibex::Ctc> outer[size];
    ibex::Array<ibex::Ctc> inner[size];
    for (auto const &ctc : contractors)
      if (ctc->get_location() < loc)
      {
        outer->add(*(ctc->get_outer()));
        inner->add(*(ctc->get_inner()));
      }

    c->set_outer(new ibex::CtcCompo(*outer));
    c->set_inner(new ibex::CtcUnion(*inner));
    c->set_location(0);
    return c;
  }

  Contractor *get_contractors()
  {
    Contractor *c = new Contractor();

    auto size = contractors.size();
    ibex::Array<ibex::Ctc> outer[size];
    ibex::Array<ibex::Ctc> inner[size];
    for (auto const &ctc : contractors)
    {
      outer->add(*(ctc->get_outer()));
      inner->add(*(ctc->get_inner()));
    }
    c->set_outer(new ibex::CtcCompo(*outer));
    c->set_inner(new ibex::CtcUnion(*inner));
    c->set_location(0);
    return c;
  }

  void dump()
  {
    std::ostringstream oss;

    for (auto const &c : contractors)
    {
      oss << "outer :" << to_oss(c->get_outer()).str() << "\n";
      oss << "inner :" << to_oss(c->get_inner()).str() << "\n";
      oss << "location :" << c->get_location() << "\n";
    }
    log_debug("contractor", "{}", oss.str());
  }
  std::ostringstream list_to_oss(ibex::Array<ibex::Ctc> *list, bool is_compo)
  {
    std::ostringstream oss;
    auto it = list->begin();
    oss << "( " << to_oss(&*it).str();
    it++;
    while (it != list->end())
    {
      oss << (is_compo ? " && " : " || ") << to_oss(&*it).str();
      it++;
    }
    oss << " )";
    return oss;
  }
  std::ostringstream to_oss(ibex::Ctc *c)
  {
    std::ostringstream oss;
    if (auto ctc_compo = dynamic_cast<ibex::CtcCompo *>(c))
      oss = list_to_oss(&ctc_compo->list, true);
    else if (auto ctc_union = dynamic_cast<ibex::CtcUnion *>(c))
      oss = list_to_oss(&ctc_union->list, false);
    else if (auto fwdbwd = dynamic_cast<ibex::CtcFwdBwd *>(c))
      oss << fwdbwd->ctr;

    return oss;
  }

  void add_contractor(ibex::Ctc *ctc, unsigned int loc)
  {
    if (ctc != nullptr)
    {
      Contractor *c = new Contractor(ctc, loc);
      contractors.insert(contractors.end(), c);
    }
  }
  bool is_empty()
  {
    return contractors.empty();
  }
};

/**
 * This class is for mapping the variables with their names and intervals.
 * It includes functionalities such as search for a variable by name and add
 * a new variable. Also update a variable interval by changing the upper/lower
 * limit.
 */
class CspMap
{
public:
  static constexpr int MAX_VAR = 20;
  static constexpr int NOT_FOUND = -1;
  static constexpr int IGNORE = -2;

  std::map<std::string, vart> var_map;

  CspMap() = default;

  size_t add_var(const std::string &name, const expr2tc &symbol)
  {
    auto find = var_map.find(name);
    if (find == var_map.end())
    {
      vart var(name, symbol, n);
      var_map.insert(std::make_pair(name, var));
      n++;
      return n - 1;
    }
    return find->second.getIndex();
  }

  void update_lb_interval(double lb, const std::string &name)
  {
    auto find = var_map.find(name);
    if (find == var_map.end())
      return;
    ibex::Interval X(lb, find->second.getInterval().ub());
    find->second.setInterval(X);
  }
  void update_ub_interval(double ub, const std::string &name)
  {
    auto find = var_map.find(name);
    if (find == var_map.end())
      return;
    ibex::Interval X(find->second.getInterval().lb(), ub);
    find->second.setInterval(X);
  }
  int find(const std::string &name)
  {
    auto find = var_map.find(name);
    if (find == var_map.end())
      return NOT_FOUND;
    return find->second.getIndex();
  }
  size_t size() const
  {
    return n;
  }

  ibex::IntervalVector create_interval_vector()
  {
    if (this->is_empty_vector)
      return ibex::IntervalVector::empty(var_map.size());

    ibex::IntervalVector X(var_map.size());
    for (auto const &var : var_map)
      X[var.second.getIndex()] = var.second.getInterval();
    return X;
  }

  void update_intervals(ibex::IntervalVector vector)
  {
    //check if interval box is empty set or if the interval is degenerated
    // in the case of a single interval
    if (vector.is_empty())
    {
      is_empty_vector = true;
      return;
    }

    for (auto &var : var_map)
    {
      if (var.second.getInterval() != vector[var.second.getIndex()])
      {
        var.second.setInterval(vector[var.second.getIndex()]);
        var.second.setIntervalChanged(true);
      }
    }
  }

private:
  size_t n = 0;
  bool is_empty_vector = false;

public:
  bool is_empty_set() const
  {
    return is_empty_vector;
  }
  void dump()
  {
    std::ostringstream oss;
    oss << "This map has : " << this->n;
    if (is_empty_vector)
    {
      oss << "empty vector";
      log_status("{}", oss.str());
      return;
    }
    log_status("{}", oss.str());

    for (auto var : var_map)
    {
      var.second.dump();
    }
  }
};
//-----------------------------------------------------------------------------------------------------------------
/// This class will parse ESBMC expressions to ibex expressions:
class expr_to_ibex_parser
{
private:
  CspMap *map;
  ibex::Variable *vars = nullptr;

  //Cleanup
  std::vector<ibex::Function *> vector_f;
  std::vector<ibex::NumConstraint *> vector_nc;
  std::vector<ibex::Ctc *> vector_ctc;

  static bool is_constraint_operator(const expr2tc &);
  static bool is_unsupported_operator_in_constraint(const expr2tc &);
  ibex::Ctc *create_contractor_from_expr2t(const expr2tc &);
  /**
   * @function create_constraint_from_expr2t is called from create_contractor_from_expr2t
   * and it will parse an expression with comparison operators and create
   * constraints to be used by create_contractor_from_expr2t.
   * @return Constraint
   */
  ibex::NumConstraint *create_constraint_from_expr2t(const expr2tc &);
  /**
   * @function create_function_from_expr2t is called by create_constraint_from_expr2t
   * and it will parse expressions with arithmetic operators and create
   * functions to be used by create_constraint_from_expr2t.
   * @return Function
   */
  //not
  ibex::Ctc *create_contractor_from_expr2t_not(const expr2tc &);
  ibex::NumConstraint *create_constraint_from_expr2t_not(const expr2tc &);

  ibex::Function *create_function_from_expr2t(expr2tc);
  int create_variable_from_expr2t(expr2tc);
  void parse_error(const expr2tc &);
  bool is_constraint_operator_not(const expr2tc &expr);
  bool is_unsupported_operator_in_constraint_not(const expr2tc &expr);

public:
  expr_to_ibex_parser(CspMap *map, ibex::Variable *vars)
  {
    this->map = map;
    this->vars = vars;
  }
  expr_to_ibex_parser() = default;
  ibex::Ctc *parse(irep_container<expr2t> expr)
  {
    return create_contractor_from_expr2t(expr);
  }
  ~expr_to_ibex_parser()
  {
    //clean up
    for (auto it = vector_ctc.rbegin(); it != vector_ctc.rend(); ++it)
      delete (*it);
    for (auto it = vector_nc.rbegin(); it != vector_nc.rend(); ++it)
      delete (*it);
    for (auto it = vector_f.rbegin(); it != vector_f.rend(); ++it)
      delete (*it);
  }
};
//-----------------------------------------------------------------------------------------------------------------
class goto_contractort : public goto_functions_algorithm
{
public:
  void goto_contractor_condition(
    const namespacet &namespacet,
    const optionst &optionst);

  /**
   * This constructor will run the goto-contractor procedure.
   * it will go through 4 steps.
   * First is parsing the properties.
   * Second, parsing the intervals.
   * Third, applying the contractor.
   * Fourth, inserting assumes in the program to reflect the contracted intervals.
   * @param _goto_functions
   */
  goto_contractort(
    goto_functionst &_goto_functions,
    const namespacet &ns,
    const optionst &options)
    : goto_functions_algorithm(true), goto_functions(_goto_functions)
  {
    if (options.get_bool_option("goto-contractor-condition"))
    {
      vars = new ibex::Variable(CspMap::MAX_VAR);
      goto_contractor_condition(ns, options);
    }
    else
    {
      initialize_main_function_loops();
      if (!function_loops.empty())
      {
        vars = new ibex::Variable(CspMap::MAX_VAR);
        parser = expr_to_ibex_parser(&map, vars);
        log_debug(
          "contractor", "1/4 - Parsing asserts to create CSP Constraints.");
        get_contractors(_goto_functions);
        if (contractors.is_empty())
        {
          log_debug(
            "contractor",
            "Contractors: expression not supported, No Contractors were "
            "created.");
          return;
        }
        contractors.dump();
        log_debug(
          "contractor",
          "2/4 - Parsing assumes to set values for variables intervals.");
        get_intervals(_goto_functions, ns);

        log_debug("contractor", "3/4 - Applying contractor.");
        apply_contractor();

        log_debug("contractor", "4/4 - Inserting assumes.");
        insert_assume(_goto_functions);
      }
    }
  }

protected:
  goto_functionst &goto_functions;

private:
  ibex::IntervalVector domains;
  ///vars variable references to be used in Ibex formulas
  ibex::Variable *vars;
  /// map is where the variable references and intervals are stored.
  CspMap map;
  /// parse ESBMC expressions to ibex
  expr_to_ibex_parser parser;
  /// contractors is where all the contractors and their complement are stored.
  Contractors contractors;

  unsigned number_of_functions = 0;

  typedef std::list<loopst> function_loopst;
  function_loopst function_loops;

  /// \Function get_contractors is a function that will go through each assert
  /// in the program and parse it from ESBMC expression to an IBEX expression
  /// that will be added create two contractors with the constraints.
  /// One is for the outer contractor with the constraint of the assert
  /// condition, another for the inner with the complement of the constraint.
  /// the function will return nothing. However the contractors be added to
  /// the list of contractors.
  /// \param functionst list of functions in the goto program
  void get_contractors(goto_functionst goto_functions);

  /// \Function get_intervals is a function that will go through each asert in
  /// the program and parse it from ESBMC expression to a triplet that are the
  /// variable name and and update its interval depending on the relation it
  /// will decide if the lower or the upper limit or both. the function will
  /// return nothing. However the values of the intervals of each variable will
  /// be updated in the Map that holds the variable information.
  /// \param functionst list of functions in the goto program
  void get_intervals(goto_functionst functionst, const namespacet &namespacet);

  /// \Function contractor function will apply the contractor on the parsed
  /// constraint and intervals. it will apply the inner contractor by
  /// calculating the complement of the assert and contract.
  /// \return Interval vector that represents the area that should be checked
  /// by the bmc.
  void apply_contractor();

  /** \Function get_complement will take a comparison operation and get its
   * complement. Operators are defined in ibex as the enumeration ibex::CmpOP.
   * @param CmpOp is a comparison operator.
   * @return complement to to received CmpOP.
   */
  ibex::CmpOp get_complement(ibex::CmpOp);
  /**
   * @function insert_assume is the function that will use the intervals
   * produced by the contractor and compare it to the original intervals.
   * If there are any changes, it will be inserted into the program as assumes
   * in the format of assume(<variable> <operator> <value>) where <variable> is
   * the variable name, <operator> is <=,>= depending if its an upper or a lower
   * limit and <value> is the value of the interval limit. If the resulting
   * interval is empty (check via is_empty_vector flag), it will insert
   * assume(0). It will also search for the last loop in the program based
   * on location.
   * @param goto_functions goto program functions
   * @param vector result from the contractor.
   */
  void insert_assume(goto_functionst goto_functions);

  void parse_intervals(expr2tc);

  bool initialize_main_function_loops();

  void insert_assume_at(
    goto_functiont goto_function,
    std::_List_iterator<goto_programt::instructiont> instruction);
};
//-----------------------------------------------------------------------------------------------------------------
class interval_analysis_ibex_contractor
{
public:
  typedef interval_templatet<BigInt> integer_intervalt;
  using real_intervalt =
    interval_templatet<boost::multiprecision::cpp_bin_float_100>;
  typedef std::unordered_map<irep_idt, integer_intervalt, irep_id_hash>
    int_mapt;

  typedef std::unordered_map<irep_idt, real_intervalt, irep_id_hash> real_mapt;

  double parse_time{}, apply_time{}, mod_time{}, cpy_time{};

  interval_analysis_ibex_contractor()
  {
    vars = new ibex::Variable(CspMap::MAX_VAR);
    map = CspMap();
    parser = expr_to_ibex_parser(&map, vars);
  }
  ~interval_analysis_ibex_contractor()
  {
    delete (vars);
  }

  bool parse_guard(expr2tc &guard)
  {
    auto t_0 = std::chrono::steady_clock::now();
    ibex::Ctc *c = parser.parse(guard);
    parse_time =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_0)
        .count();

    if (c == nullptr)
      return false;
    else if (map.var_map.size() == 0)
      return false;

    contractor = Contractor(c);
    return true;
  }

  void maps_to_domains(int_mapt, real_mapt);

  void apply_contractor();

  expr2tc result_of_outer(expr2tc exp);

  void dump();

  [[maybe_unused]] void modularize_intervals();

private:
  ibex::IntervalVector domains;
  ///vars variable references to be used in Ibex formulas
  ibex::Variable *vars;
  /// map is where the variable references and intervals are stored.
  CspMap map;
  CspMap map_outer;
  /// parser from expr to ibex
  expr_to_ibex_parser parser;
  /// contractor is where the contractor is stored.
  Contractor contractor{};

  void _dump()
  {
    std::ostringstream oss;
    auto c = contractor;
    oss << "constraint :" << to_oss(c.get_outer()).str();
    log_debug("contractor", "{}", oss.str());
  }
  std::ostringstream list_to_oss(ibex::Array<ibex::Ctc> *list, bool is_compo)
  {
    std::ostringstream oss;
    auto it = list->begin();
    oss << "( " << to_oss(&*it).str();
    it++;
    while (it != list->end())
    {
      oss << (is_compo ? " && " : " || ") << to_oss(&*it).str();
      it++;
    }
    oss << " )";
    return oss;
  }
  std::ostringstream to_oss(ibex::Ctc *c)
  {
    std::ostringstream oss;
    if (auto ctc_compo = dynamic_cast<ibex::CtcCompo *>(c))
      oss = list_to_oss(&ctc_compo->list, true);
    else if (auto ctc_union = dynamic_cast<ibex::CtcUnion *>(c))
      oss = list_to_oss(&ctc_union->list, false);
    else if (auto fwdbwd = dynamic_cast<ibex::CtcFwdBwd *>(c))
      oss << fwdbwd->ctr;

    return oss;
  }
};
#endif //ESBMC_GOTO_CONTRACTOR_H