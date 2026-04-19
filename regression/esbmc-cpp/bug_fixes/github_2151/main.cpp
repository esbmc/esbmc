#include <iostream> // Inclui a biblioteca de entrada e saída padrão
#include <vector>   // Inclui a biblioteca de vetores
#include <map>      // Inclui a biblioteca de mapas (associações chave-valor)
#include <string>   // Inclui a biblioteca de strings
#include <tuple> // Inclui a biblioteca de tuplas para usar make_tuple e tuple
#include <cassert> // Inclui a biblioteca de assertivas para verificação de condições

using namespace std; // Usa o namespace padrão para evitar a necessidade de prefixar std::

// Definição das variáveis e funções de transição de estado

// Conjuntos
vector<string> C = {"motor", "chassi", "rodas"};        // Componentes
vector<string> L = {"estacao1", "estacao2", "armazem"}; // Locais
int T = 10; // Instantes de tempo discretos

// Tempos médios em cada estação de trabalho (em unidades de tempo)
map<string, int> tempos_medios = {
  {"estacao1", 3},
  {"estacao2", 4},
  {"armazem", 2}};

// Tempo total de ciclo esperado
int tempo_total_ciclo = 9;

// Variáveis
map<tuple<string, string, int>, int>
  x; // Mapa para rastrear a localização dos componentes ao longo do tempo
map<tuple<string, int>, int>
  r; // Mapa para rastrear leituras RFID dos componentes ao longo do tempo
map<tuple<string, int>, string>
  s; // Mapa para rastrear o status dos componentes ao longo do tempo
map<string, int>
  tempo_em_estacao; // Mapa para rastrear o tempo gasto em cada estação por componente
map<string, int>
  tempo_total; // Mapa para rastrear o tempo total de ciclo por componente

// Inicialização das variáveis
void inicializa_variaveis()
{
  for (const auto &c : C)
  { // Para cada componente
    for (const auto &l : L)
    { // Para cada local
      for (int t = 0; t < T; ++t)
      { // Para cada instante de tempo
        x[make_tuple(c, l, t)] =
          0; // Inicializa a localização como 0 (não presente)
      }
    }
    for (int t = 0; t < T; ++t)
    {                          // Para cada instante de tempo
      r[make_tuple(c, t)] = 0; // Inicializa a leitura RFID como 0 (não lido)
      s[make_tuple(c, t)] = "IDLE"; // Inicializa o status como "IDLE" (inativo)
    }
    tempo_em_estacao[c] = 0; // Inicializa o tempo em estação como 0
    tempo_total[c] = 0;      // Inicializa o tempo total como 0
  }
}

// Função de Leitura RFID
int leitura_rfid(const string &c, int t)
{
  // Simulação de leitura RFID
  return (t % 2 == 0)
           ? 1
           : 0; // Retorna 1 se o tempo for par, caso contrário retorna 0
}

// Função de Atualização de Localização
void atualiza_localizacao(const string &c, const string &l, int t)
{
  if (r[make_tuple(c, t)] == 1)
  { // Se a leitura RFID for 1
    // Primeiro, zere a localização do componente em todas as estações
    for (const auto &loc : L)
    {
      x[make_tuple(c, loc, t + 1)] = 0;
    }
    // Depois, marque a nova localização
    x[make_tuple(c, l, t + 1)] = 1;
    tempo_em_estacao[c]++; // Incrementa o tempo em estação para o componente
  }
  else
  {
    x[make_tuple(c, l, t + 1)] = 0; // Mantém a localização como 0
  }
}

// Função de Atualização de Status
void atualiza_status(const string &c, int t)
{
  if (r[make_tuple(c, t)] == 1)
  { // Se a leitura RFID for 1
    s[make_tuple(c, t + 1)] =
      "PROC"; // Atualiza o status para "PROC" (em processamento)
  }
  else
  {
    s[make_tuple(c, t + 1)] = "IDLE"; // Mantém o status como "IDLE" (inativo)
  }
}

// Função para sinalizar a situação de produção
void sinalizar_situacao(const string &c)
{
  if (tempo_total[c] < tempo_total_ciclo)
  { // Se o tempo total for menor que o tempo de ciclo esperado
    cout << "Componente " << c << ": Adiantado"
         << endl; // Sinaliza adiantamento
  }
  else if (tempo_total[c] == tempo_total_ciclo)
  { // Se o tempo total for igual ao tempo de ciclo esperado
    cout << "Componente " << c << ": Normal" << endl; // Sinaliza normalidade
  }
  else
  { // Se o tempo total for maior que o tempo de ciclo esperado
    cout << "Componente " << c << ": Atrasado" << endl; // Sinaliza atraso
  }
}

// Implementação do Algoritmo
void simula_transicao_estados()
{
  for (int t = 0; t < T; ++t)
  { // Para cada instante de tempo
    for (const auto &c : C)
    {                                           // Para cada componente
      r[make_tuple(c, t)] = leitura_rfid(c, t); // Atualiza a leitura RFID
      for (const auto &l : L)
      {                                // Para cada local
        atualiza_localizacao(c, l, t); // Atualiza a localização
      }
      atualiza_status(c, t); // Atualiza o status
    }
  }
}

// Exibir resultados
void exibir_resultados()
{
  for (int t = 0; t < T; ++t)
  { // Para cada instante de tempo
    for (const auto &c : C)
    { // Para cada componente
      cout << "Tempo " << t << ": Componente " << c << " - Localização: [";
      for (const auto &l : L)
      { // Para cada local
        cout << x[make_tuple(c, l, t)]
             << " "; // Exibe a localização do componente
      }
      cout << "], Status: " << s[make_tuple(c, t)]
           << endl; // Exibe o status do componente
    }
  }
  for (const auto &c : C)
  {                                       // Para cada componente
    tempo_total[c] = tempo_em_estacao[c]; // Calcula o tempo total de ciclo
    sinalizar_situacao(c);                // Sinaliza a situação de produção
  }
}

// Função principal
int main()
{
  inicializa_variaveis(); // Inicializa as variáveis

  // Verificar Inicialização das Variáveis
  assert(x[make_tuple("motor", "estacao1", 0)] == 0);
  assert(r[make_tuple("motor", 0)] == 0);
  assert(s[make_tuple("motor", 0)] == "IDLE");

  simula_transicao_estados(); // Simula a transição de estados

  // Verificar Leitura RFID
  assert(leitura_rfid("motor", 0) == 1);
  assert(leitura_rfid("motor", 1) == 0);

  // Verificar Atualização de Localização
  atualiza_localizacao("motor", "estacao1", 0);
  assert(x[make_tuple("motor", "estacao1", 1)] == 1);

  atualiza_status("motor", 0);
  assert(s[make_tuple("motor", 1)] == "PROC");

  assert(r[make_tuple("motor", 0)] == 1);
  assert(x[make_tuple("motor", "estacao1", 1)] == 1);
  assert(s[make_tuple("motor", 1)] == "PROC");

  exibir_resultados();
  assert(x[make_tuple("motor", "estacao1", 9)] == 1);
  assert(s[make_tuple("motor", 9)] == "PROC");

  for (int t = 0; t < T; ++t)
  {
    for (const auto &c : C)
    {
      assert(
        x[make_tuple(c, "estacao1", t)] == 0 ||
        x[make_tuple(c, "estacao2", t)] == 0 ||
        x[make_tuple(c, "armazem", t)] == 0);
    }
  }

  return 0;
}
