import cv2
import numpy as np
import matplotlib.pyplot as plt

def mostrar_hue_e_mascara(img_path):
    # 1. Carrega a imagem
    img = cv2.imread(img_path)
    if img is None:
        print("Erro ao carregar a imagem.")
        return

    # 2. Converte para HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]  # Apenas o canal H

    # 3. Define a máscara para tons de vermelho
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([165, 70, 50])
    upper_red2 = np.array([179, 255, 255])
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    # 4. Aplica a máscara à imagem original
    img_mascarada = cv2.bitwise_and(img, img, mask=mask)

    # 5. Converte imagens para RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_mascarada_rgb = cv2.cvtColor(img_mascarada, cv2.COLOR_BGR2RGB)

    # 6. Normaliza o canal Hue para visualização com colormap (HSV → RGB visual)
    hue_colormap = cv2.applyColorMap(cv2.normalize(hue, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_HSV)
    hue_colormap_rgb = cv2.cvtColor(hue_colormap, cv2.COLOR_BGR2RGB)

    # 7. Visualização em 4 partes
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Imagem Original")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(hue, cmap='gray', vmin=0, vmax=179)
    plt.title("Canal Hue (Matiz) em Tons de Cinza")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(hue_colormap_rgb)
    plt.title("Hue Mapeado para Cores (Visual)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(img_mascarada_rgb)
    plt.title("Pixels Capturados pela Máscara")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
  mostrar_hue_e_mascara("recorte/colcheia.png")
  mostrar_hue_e_mascara("recorte/colcheia-rosada.png")
  mostrar_hue_e_mascara("recorte/colcheia-interacao.png")
  mostrar_hue_e_mascara("recorte/colcheia-clara.png")
  mostrar_hue_e_mascara("recorte/colcheia-escura.png")
