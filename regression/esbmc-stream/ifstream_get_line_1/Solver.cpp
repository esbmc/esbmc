/*
 * Solver.cpp
 *
 *  Created on: Jul 24, 2012
 *      Author: mikhail
 */

#include "Solver.h"

#include <cmath>

Solver::Solver(int in_num_of_vars, int in_num_of_clauses,	vector<Clause> in_clauses)
  : num_of_vars(in_num_of_vars),
    num_of_clauses(in_num_of_clauses),
    num_of_cases(pow(2.0,in_num_of_vars)),
    clauses(in_clauses)
{
	cout << "============================================================================================" << endl;
	cout << "Numero de variaveis: " << num_of_vars << endl;
	cout << "Numero de clausulas: " << num_of_clauses << endl;
	cout << "Numero de casos da tabela verdade: " << num_of_cases << endl;
	cout << "============================================================================================" << endl << endl;

	cria_tabela_verdade();

}

void Solver::cria_tabela_verdade()
{
	tabela_verdade.resize(num_of_cases);

	for (int i=0; i<num_of_cases; ++i)
		tabela_verdade.at(i).resize(num_of_vars);

	int aux = num_of_cases/2;
	int aux1 = 1;
	bool inverte = true;

	for(int j=0; j<num_of_vars; ++j)
	{
		for(int i=0; i<num_of_cases; ++i)
		{
			if(inverte)
			{
				tabela_verdade.at(i).at(j) = true;
			}

			if(aux1 == aux)
			{
				inverte = !inverte;
				aux1 = 1;
			}
			else
				++aux1;
		}

		aux1 = 1;
		aux = aux/2;
	}

//	cout << "altura " << tabela_verdade.size() << endl;
//	for (int i=0; i<num_of_cases; ++i)
//		cout << "largura " << tabela_verdade.at(i).size() << endl;

//	for(unsigned int i=0; i<tabela_verdade.size(); ++i) {
//		for(unsigned int j=0; j<tabela_verdade.at(i).size(); ++j)
//			cout << tabela_verdade.at(i).at(j) << "\t";
//		cout << endl;
//	}
}

void Solver::solve()
{
	vector<int> solucoes;

	for(unsigned int i=0; i<tabela_verdade.size(); ++i)
	{
		bool operacao_and = true;

		for(unsigned int j=0; j<clauses.size(); ++j)
		{
			bool operacao_or = false;

			for(unsigned int k=0; k<clauses.at(j).variables.size(); ++k)
			{
				int var = clauses.at(j).variables.at(k);

				if(var > 0)
					operacao_or |= tabela_verdade.at(i).at(var-1);
				else
					operacao_or |= !tabela_verdade.at(i).at((-1*var)-1);
			}

			operacao_and &= operacao_or;
		}

		if(operacao_and)
			solucoes.push_back(i);
	}

	escreve_saida(solucoes);
}

void Solver::escreve_saida(vector<int> solucoes)
{
	cout << "Encontradas " << solucoes.size() << " solucoes: " << endl;

	for(int i=0; i<num_of_vars; ++i)
		cout << "x(" << i+1 << ")\t";
	cout << endl;

	for(unsigned int i=0; i<solucoes.size(); ++i)
	{
		int linha = solucoes.at(i);
		for(int j=0; j<num_of_vars; ++j)
		{
			cout << tabela_verdade.at(linha).at(j) << "\t";
		}
		cout << endl;
	}

	cout << "============================================================================================" << endl;
}

void Solver::solveMultithread()
{
	// TODO
}
