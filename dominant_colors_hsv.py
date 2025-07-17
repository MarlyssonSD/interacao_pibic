import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

def encontrar_hsv_dominantes(img_path, num_cores=5):
    img = cv2.imread(img_path)

    if img is None: return []

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_mask = np.array([0, 20, 20]); upper_mask = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_img, lower_mask, upper_mask)
    pixels = hsv_img[mask > 0]

    if len(pixels) < num_cores: return []

    kmeans = KMeans(n_clusters=num_cores, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    
    return kmeans.cluster_centers_.astype(np.uint8)

def calibrar_mascara_universal(lista_imagens_amostra, margem=15):
    """
    Função de calibração aprimorada que detecta o wraparound do Hue.
    Retorna uma LISTA de pares de limites. Ex: [(lower1, upper1)] ou [(lower1, upper1), (lower2, upper2)]
    """
    todas_as_cores_hsv = []
    print("--- Passo 1: Analisando imagens de amostra para coletar cores ---")
    for img_path in lista_imagens_amostra:
        print(f"Analisando '{img_path}'...")
        cores = encontrar_hsv_dominantes(img_path)
        if len(cores) > 0: todas_as_cores_hsv.extend(cores)
    
    if not todas_as_cores_hsv:
        print("ERRO: Nenhuma cor dominante foi encontrada."); return []

    hsv_array = np.array(todas_as_cores_hsv)
    
    # --- INÍCIO DA NOVA LÓGICA DE WRAPAROUND ---
    
    hues = hsv_array[:, 0]
    sats = hsv_array[:, 1]
    vals = hsv_array[:, 2]

    h_min, h_max = np.min(hues), np.max(hues)
    s_min, s_max = np.min(sats), np.max(sats)
    v_min, v_max = np.min(vals), np.max(vals)

    # Heurística para detectar o wraparound: se a faixa de Hue for muito grande
    if (h_max - h_min) > 100: # O valor 100 pode ser ajustado. 180/2=90, então >90 é um bom sinal.
        print("\n--- Detectado 'Wraparound' de Hue! Gerando duas máscaras. ---")
        
        # Separa os hues em dois grupos: baixos (perto de 0) e altos (perto de 179)
        # O ponto de corte 90 é seguro (verde/ciano)
        hues_low = hues[hues < 90]
        hues_high = hues[hues > 90]
        
        # Limite 1 (vermelhos baixos)
        lower1 = np.array([0, s_min, v_min])
        upper1 = np.array([np.max(hues_low), s_max, v_max])
        
        # Limite 2 (vermelhos altos)
        lower2 = np.array([np.min(hues_high), s_min, v_min])
        upper2 = np.array([179, s_max, v_max])
        
        # Adiciona margem de segurança
        lower1 = np.clip(lower1.astype(int) - margem, 0, 255).astype(np.uint8)
        upper1 = np.clip(upper1.astype(int) + margem, 0, 255).astype(np.uint8)
        lower2 = np.clip(lower2.astype(int) - margem, 0, 255).astype(np.uint8)
        upper2 = np.clip(upper2.astype(int) + margem, 0, 255).astype(np.uint8)
        
        # Garante que o limite superior de Hue não passe de 179
        upper1[0] = min(upper1[0], 179)
        upper2[0] = min(upper2[0], 179)

        print(f"Limites FINAIS 1 (Inferior): {lower1}")
        print(f"Limites FINAIS 1 (Superior): {upper1}")
        print(f"Limites FINAIS 2 (Inferior): {lower2}")
        print(f"Limites FINAIS 2 (Superior): {upper2}")
        
        return [(lower1, upper1), (lower2, upper2)]
        
    else: # Caso normal, sem wraparound
        print("\n--- Nenhum 'Wraparound' detectado. Gerando uma única máscara. ---")
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # Adiciona margem de segurança
        lower = np.clip(lower.astype(int) - margem, 0, 255).astype(np.uint8)
        upper = np.clip(upper.astype(int) + margem, 0, 255).astype(np.uint8)
        upper[0] = min(upper[0], 179)
        
        print(f"Limites FINAIS (Inferior): {lower}")
        print(f"Limites FINAIS (Superior): {upper}")

        return [(lower, upper)]
        
def aplicar_e_mostrar_mascara(img_path, lista_de_limites):
    """Aplica uma ou mais máscaras a uma imagem e mostra o resultado."""
    img = cv2.imread(img_path)
    if img is None: return
        
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Cria uma máscara vazia inicial
    mascara_final = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    
    # Itera sobre a lista de limites (pode ter 1 ou 2 pares)
    for lower, upper in lista_de_limites:
        mascara_parcial = cv2.inRange(hsv_img, lower, upper)
        # Combina a máscara parcial com a máscara final
        mascara_final = cv2.bitwise_or(mascara_final, mascara_parcial)

    # Limpeza opcional da máscara
    kernel = np.ones((5,5),np.uint8)
    mascara_final = cv2.morphologyEx(mascara_final, cv2.MORPH_CLOSE, kernel)
    mascara_final = cv2.morphologyEx(mascara_final, cv2.MORPH_OPEN, kernel)
    
    resultado = cv2.bitwise_and(img, img, mask=mascara_final)
    
    img_comparacao = np.hstack((img, resultado))
    plt.figure(figsize=(12,6))
    plt.imshow(cv2.cvtColor(img_comparacao, cv2.COLOR_BGR2RGB))
    plt.title(f'Original vs. Máscara Universal em "{img_path}"')
    plt.axis('off'); plt.show()

# --- BLOCO PRINCIPAL DE EXECUÇÃO ---
if __name__ == "__main__":
    pasta_de_amostras = "imagens_cores"
    extensoes_validas = ('.png', '.jpg', '.jpeg')

    imagens_de_amostra = [
        os.path.join(pasta_de_amostras, arquivo) 
        for arquivo in os.listdir(pasta_de_amostras) 
        if arquivo.lower().endswith(extensoes_validas)
    ]

    limites = calibrar_mascara_universal(imagens_de_amostra, margem=15)

    # if limites:
    #     for img_path in imagens_de_amostra:
            # aplicar_e_mostrar_mascara(img_path, limites)