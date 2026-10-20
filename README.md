```
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import SAM

# 1. Carregar o modelo SAM pré-treinado
# O modelo 'sam_b.pt' é a versão base, ideal para testes rápidos no Colab
model = SAM("sam_b.pt")

# 2. Definir o caminho da sua imagem contendo a pluma de fumaça
# Substitua pelo caminho correto da imagem no seu ambiente
image_path = "caminho_para_sua_imagem_de_fumaca.jpg"

# Carregar a imagem com OpenCV (conversão para RGB para exibição correta)
image_rgb = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
h, w, _ = image_rgb.shape

# ==========================================
# CONFIGURAÇÃO 1: Apenas Bounding Box (Caixa Delimitadora)
# ==========================================
# Defina as coordenadas [x_min, y_min, x_max, y_max] que cercam a pluma de fumaça
bbox_coords = [
    100,
    50,
    400,
    300,
]  # Ajuste esses valores de acordo com a sua imagem!

results_box = model(image_rgb, bboxes=[bbox_coords])
mask_box = results_box[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255

# ==========================================
# CONFIGURAÇÃO 2: Bounding Box + Pontos Positivos
# ==========================================
# Pontos em cima da fumaça (coordenadas [x, y]) indicando o que incluir (rótulo 1)
points_pos = [[250, 150], [220, 120]]
labels_pos = [1, 1]

results_pos = model(
    image_rgb,
    bboxes=[bbox_coords],
    points=points_pos,
    labels=labels_pos,
)
mask_pos = results_pos[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255

# ==========================================
# CONFIGURAÇÃO 3: Bounding Box + Pontos Positivos + Pontos Negativos
# ==========================================
# Adicionando pontos negativos (rótulo 0) em nuvens ou céu que o modelo pegou por engano
points_mixed = [[250, 150], [220, 120], [50, 30]]  # O último ponto é no fundo/céu
labels_mixed = [1, 1, 0]  # 0 indica fundo/exclusão

results_mixed = model(
    image_rgb,
    bboxes=[bbox_coords],
    points=points_mixed,
    labels=labels_mixed,
)
mask_mixed = (
    results_mixed[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
)


# ==========================================
# FUNÇÃO DE PLOTAGEM (Lado a Lado: Original, Binária, Sobreposta)
# ==========================================
def plot_config(original_img, binary_mask, title):
  # Criar imagem sobreposta (com transparência vermelha na máscara)
  overlay = original_img.copy()
  overlay[binary_mask > 0] = [
      255,
      0,
      0,
  ]  # Pinta de vermelho onde a máscara for ativa
  blended = cv2.addWeighted(original_img, 0.7, overlay, 0.3, 0)

  fig, axes = plt.subplots(1, 3, figsize=(15, 5))

  axes[0].imshow(original_img)
  axes[0].setTitle("Imagem Original")
  axes[0].axis("off")

  axes[1].imshow(binary_mask, cmap="gray")
  axes[1].setTitle("Máscara Binária")
  axes[1].axis("off")

  axes[2].imshow(blended)
  axes[2].setTitle("Sobreposta à Imagem")
  axes[2].axis("off")

  plt.suptitle(title, fontsize=16)
  plt.tight_layout()
  plt.show()


# Executando a visualização para as 3 configurações
plot_config(image_rgb, mask_box, "Configuração 1: Apenas Bounding Box")
plot_config(
    image_rgb,
    mask_pos,
    "Configuração 2: Bounding Box + Pontos Positivos",
)
plot_config(
    image_rgb,
    mask_mixed,
    "Configuração 3: Bounding Box + Pontos Positivos e Negativos",
)
```
##Análise dos Resultados e Justificativas
1. Avaliação das Máscaras Geradas e Desempenho dos PromptsConfiguração 1 (Apenas Bounding Box):
   Apenas com a caixa delimitadora, o modelo capturou bem a região superior mais densa da pluma de fumaça, porém deixou de fora a base do incêndio (próxima à vegetação) e as áreas de menor densidade/transparência.   Configuração 2 (Bounding Box + Pontos Positivos): A inclusão dos pontos positivos buscou expandir a segmentação para cobrir mais fumaça, mas gerou um "falso positivo" significativo: o modelo acabou incluindo regiões de névoa/bruma no céu ao lado esquerdo, devido à semelhança de tonalidades.   Configuração 3 (Bounding Box + Pontos Positivos e Negativos): A inclusão de pontos negativos no céu e na névoa ao lado esquerdo foi fundamental para limitar a área de seleção. Isso eliminou o excesso capturado na Configuração 2 e refinou os contornos da pluma de fumaça real.
2. Dificuldades Encontradas e Impacto das Características da ImagemIluminação e Contraste: O tom cinza e esbranquiçado da fumaça no topo se confunde com o fundo claro e a bruma atmosférica do horizonte. Essa baixa variação de contraste dificulta para o algoritmo a identificação exata de onde a fumaça termina e a atmosfera começa. A presença de névoa e poeira sobre a vegetação ao fundo possui características de cor e textura muito parecidas com as da fumaça difusa. Sem a orientação de pontos negativos, o modelo tende a interpretar esses elementos como parte do mesmo objeto. Diferente de objetos rígidos (como edifícios ou carros) que possuem contornos bem definidos, a fumaça apresenta limites gradientes e transparência variável. Isso torna a definição de uma máscara binária (preto e branco) um desafio complexo para o modelo.
  <img width="1489" height="437" alt="download (1)" src="https://github.com/user-attachments/assets/0702fcca-095b-4e1c-aac0-6616dc60f6e8"/> 
  <img width="1489" height="437" alt="download (2)" src="https://github.com/user-attachments/assets/78329e92-6779-4722-9a1d-da7726de6ed8" />
  <img width="1489" height="437" alt="download (3)" src="https://github.com/user-attachments/assets/d0f0f070-0596-4116-a782-4cf942aec53a" />


