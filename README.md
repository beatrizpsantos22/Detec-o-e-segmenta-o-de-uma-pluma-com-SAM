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
