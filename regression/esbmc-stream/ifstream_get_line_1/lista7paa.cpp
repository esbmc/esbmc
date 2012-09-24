#include <iostream>
using namespace std;

#include "Parser.h"
#include "Solver.h"

int main(int argc, const char **argv)
{
	if(argc != 2) {
		cout << "Erro! Arquivo no formato cnf deve ser passado como parametro" << endl;
		return -1;
	}

	try
	{
		Parser p(argv[1]);
//		p.printClauses();
//		p.printEquation();

		Solver s(p.num_of_vars, p.num_of_clauses, p.clauses);
		s.solve();
	}
	catch(int e)
	{
		switch(e)
		{
			case 0:
				cout << "Erro! Não consegui abrir o arquivo." << endl;
				break;

			case 1:
				cout << "Erro! Falha no parser linha p." << endl;
//				cout << "Linha p deve seguir o formato: \"p cnf num_of_vars num_of_clause \(num_of_vars e num_of_clauses devem ser > 1)\"" << endl;
				break;

			case 2:
				cout << "Erro! Clausula inicou sem definicao do problema." << endl;
				break;

			case 3:
				cout << "Erro! Linha p definida duas vezes." << endl;
				break;

			case 4:
				cout << "Erro! Clausulas faltando." << endl;
				break;

			case 5:
				cout << "Erro! Clausulas numeradas erroneamente." << endl;
				break;
		}
	}

	return 0;
}
