import numpy as np
from sklearn.metrics import r2_score

# 1. Seus dados de entrada (Nascimentos totais no Brasil)
x = np.array([1, 2 , 3, 4, 5, 6, 7, 8, 9, 10, 11])
y = np.array([3017668, 2857800, 2923535, 2944932, 2849146, 2730145, 2677101, 2561922, 2537576, 2389325, 2456688])


# 2. Configurações de busca
melhor_r2 = -float('inf')
melhor_grau = 1
melhor_modelo = None

# Max_grau não pode ser maior que (número de pontos - 1)
max_grau = min(10, len(x) - 1) 

# 3. Loop para testar os graus e encontrar o melhor
for grau in range(1, max_grau + 1):
    # Ajusta o polinômio para o grau atual
    coeficientes = np.polyfit(x, y, grau)
    modelo = np.poly1d(coeficientes)
    
    # Calcula as previsões e o R²
    y_previsto = modelo(x)
    r2 = r2_score(y, y_previsto)
    
    # Atualiza se encontrar um R² melhor
    if r2 > melhor_r2:
        melhor_r2 = r2
        melhor_grau = grau
        melhor_modelo = modelo

# 4. Resultados
print(f"🏆 Melhor Grau Encontrado: {melhor_grau}")
print(f"📊 R² do Modelo: {melhor_r2:.4f}")
print("\n📝 Equação do Polinômio:")
print(melhor_modelo)