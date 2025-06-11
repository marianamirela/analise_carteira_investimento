# MODELO DE OTIMIZAÇÃO COM ATIVO LIVRE DE RISCO E VENDA A DESCOBERTO

# Importar bibliotecas necessárias
    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt  

# Definição de função objetivo
    def teta(x):
        retorno_excesso = np.dot((RM_ativos - retorno_livre_risco).T, x)
        risco = np.dot(x.T, np.dot(Q, x))**(1/2)
        return -retorno_excesso/risco
    
# Definição de restrição de soma dos pesos igual a 1
    def restricao(x):
        return np.sum(x) - 1

# Gerar retornos sintéticos
    np.random.seed(42)
    retornos = np.random.normal(loc=0.001, scale=0.02, size=(252, 5))
    livre = np.random.normal(loc=0.0001, scale=0.0005, size=252)  

# Parâmetros
    RM_ativos = np.mean(retornos, axis=0)
    Q = np.cov(retornos.T)
    retorno_livre_risco = np.mean(livre)

# Chute inicial
    x0 = np.ones(5)/5

# Restrições e fronteiras
    restricoes = ({’type’: ’eq’, ’fun’: restricao})

# Otimização
    sol = minimize(teta, x0, constraints=restricoes)
    x_tangente = sol.x
    vteta = -sol.fun
    Sigma_t = np.dot(x_tangente.T, np.dot(Q, x_tangente))**(1/2)
    R_t = np.dot(RM_ativos.T, x_tangente)

# Reta tangente
    Sigma = np.linspace(0, 0.06, 100)
    R = retorno_livre_risco + vteta*Sigma

# Fronteira eficiente com venda a descoberto
      Retornos_carteira = []
      Riscos_carteira = []
      for k in np.linspace(min(RM_ativos), max(RM_ativos), 100):
          restricoes_k = ({’type’: ’eq’, ’fun’: restricao},
                          {’type’: ’eq’, ’fun’: lambda x, k=k: np.dot(RM_ativos.T, x) - k})
          sol = minimize(lambda x: np.dot(x.T, np.dot(Q, x)), x0, constraints=restricoes_k)
          Retornos_carteira.append(k)
          Riscos_carteira.append(np.sqrt(sol.fun))

# Plot
    plt.figure(figsize=(10, 6))
    plt.plot(Riscos_carteira, Retornos_carteira, label=’Fronteira Eficiente (CVD)’)
    plt.plot(Sigma, R, label=’Reta Tangente’, linestyle=’--’)
    plt.scatter(Sigma_t, R_t, color=’red’, label=’Carteira Tangente’)
    plt.title(’Fronteira Eficiente com Ativo Livre de Risco (Venda a Descoberto)’)
    plt.xlabel(’Risco (Desvio Padrão)’)
    plt.ylabel(’Retorno Esperado’)
    plt.legend()
    plt.grid(True)
    plt.show()

