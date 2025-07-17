import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import view_masks as vm
import dominant_colors_hsv as dch

# Aumente fator_vermelho para ser mais seletivo (ex: 1.5).

# Diminua fator_verde para aceitar menos verde (ex: 0.5).

# Aumente limiar_vermelho para ignorar vermelhos fracos (ex: 150).

# Pode criar mais filtros (ex: azul < média * 0.8) para tirar interferências.
def saturar_vermelho(imagem_rgb, fator_vermelho=1.2, fator_verde=0.8, limiar_vermelho=100):
    # imagem_rgb: numpy array RGB
    r = imagem_rgb[:, :, 0].astype(np.float32)
    g = imagem_rgb[:, :, 1].astype(np.float32)
    b = imagem_rgb[:, :, 2].astype(np.float32)
    
    media = (r + g + b) / 3

    # Máscara para pixels que são vermelhos intensos segundo os critérios
    mask = (r > media * fator_vermelho) & (g < media * fator_verde) & (r > limiar_vermelho)

    # Cria imagem preta
    img_saida = np.zeros_like(imagem_rgb)
    # Aplica vermelho saturado só onde a máscara é True
    img_saida[mask] = [255, 0, 0]

    return img_saida



def dissimilaridade_cos(hist1, hist2):
    hist1 = hist1.flatten()
    hist2 = hist2.flatten()
    numerador = np.dot(hist1, hist2)
    denominador = np.linalg.norm(hist1) * np.linalg.norm(hist2)
    print(f"Numerador: {numerador}, Denominador: {denominador}")
    if denominador == 0:
        return 1  # dissimilaridade máxima
    return 1 - (numerador / denominador) #inversão do valor da similaridade


def analisar_e_plotar_comparacao(img_base_path, img_comp_path, threshold=0.6):
    img_base = cv2.imread(img_base_path)
    img_comp = cv2.imread(img_comp_path)

    if img_base is None or img_comp is None:
        print(f"ERRO: Não foi possível carregar as imagens.")
        return

    # Converte para HSV
    rgb_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
    rgb_comp = cv2.cvtColor(img_comp, cv2.COLOR_BGR2RGB)

    rgb_base_sat = saturar_vermelho(rgb_base)
    rgb_comp_sat = saturar_vermelho(rgb_comp)

    # Máscaras para vermelho em hsv
    # lower_red1 = np.array([0, 20, 20])
    # upper_red1 = np.array([15, 255, 255])
    # lower_red2 = np.array([165, 20, 20])
    # upper_red2 = np.array([179, 255, 255])
    lower_red = np.array([150, 0, 0])  # vermelho mais escuro
    upper_red = np.array([255, 100, 100])  # tons de vermelho mais claro

    mask_base = cv2.inRange(rgb_base_sat, lower_red, upper_red)
    mask_comp = cv2.inRange(rgb_comp_sat, lower_red, upper_red)


    # Calculo e normalização dos histogramas
    # hist_base = cv2.calcHist([hsv_base], [0], mask_base, [180], [0, 180])
    # hist_comp = cv2.calcHist([hsv_comp], [0], mask_comp, [180], [0, 180])
    hist_base = cv2.calcHist([rgb_base], [0], mask_base, [256], [0, 256])
    hist_comp = cv2.calcHist([rgb_comp], [0], mask_comp, [256], [0, 256])


    # RGB

    # rgb_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
    # rgb_comp = cv2.cvtColor(img_comp, cv2.COLOR_BGR2RGB)

    # lower_red = np.array([150, 0, 0])  # vermelho mais escuro
    # upper_red = np.array([255, 100, 100])  # tons de vermelho mais claro

    # mask_base = cv2.inRange(rgb_base, lower_red, upper_red)
    # mask_comp = cv2.inRange(rgb_comp, lower_red, upper_red)


    cv2.normalize(hist_base, hist_base, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_comp, hist_comp, 0, 1, cv2.NORM_MINMAX)

    cv2.imshow("Máscara base", mask_base)
    cv2.imshow("Máscara comparação", mask_comp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Comparação de histogramas
    score = dissimilaridade_cos(hist_base, hist_comp)
    interacao = score > threshold
    
    print(f"Analisando '{img_base_path}' vs '{img_comp_path}'...")
    print(f"Score de disimilaridade Calculado: {score:.4f}")
    print(f"O score > {threshold}? {'Sim' if interacao else 'Não'}")
    print(f"Resultado Final: {'Interação com alvo detectada' if interacao else 'Não houve interação com o alvo'}")

    # Plotar os resultados
    img_base_rgb = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
    img_comp_rgb = cv2.cvtColor(img_comp, cv2.COLOR_BGR2RGB)
    
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    resultado_texto = f"Resultado: {'Interação com alvo detectada' if interacao else 'Não houve interação com o alvo'}"
    score_texto = f"Score de disimilaridade: {score:.4f}"
    fig.suptitle(f"{resultado_texto} | {score_texto}", fontsize=16, weight='bold')

    axs[0, 1].plot(hist_base, color='red')
    axs[0, 1].set_title("Histograma do Vermelho - Base")
    axs[0, 1].set_xlim([0, 256]); axs[0, 1].grid(True, alpha=0.5)

    axs[1, 1].plot(hist_comp, color='blue')
    axs[1, 1].set_title("Histograma do Vermelho - Comparação")
    axs[1, 1].set_xlim([0, 256]); axs[1, 1].grid(True, alpha=0.5)


    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # salvar o gráfico em uma pasta
    nome_base = os.path.splitext(os.path.basename(img_comp_path))[0]
    output_dir = "comparacoes_salvas_coss_RGB"
    os.makedirs(output_dir, exist_ok=True)
    caminho_salvar = os.path.join(output_dir, f"comparacao_{nome_base}.png")
    plt.savefig(caminho_salvar)
    print(f"Imagem salva em: {caminho_salvar}")
    
    # plt.show()

def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear') 

def menu():

    while True:
        # limpar_tela()
        print("=========================================")
        print("   MENU   ")
        print("=========================================")
        print("1. Teste de Interação")
        print("2. Visualização das cores presentes na máscara")
        print("-----------------------------------------")
        print("0. Sair")
        print("=========================================")
        
        escolha = input("Digite a opção desejada: ")

        if escolha == '1':
            analisar_e_plotar_comparacao(
                img_base_path="colcheia/base.png", 
                img_comp_path="colcheia/interacao.png"
            )

            analisar_e_plotar_comparacao(
                img_base_path="colcheia/base.png", 
                img_comp_path="colcheia/teste1.png"
            )
            analisar_e_plotar_comparacao(
                img_base_path="colcheia/base.png", 
                img_comp_path="colcheia/teste2.png"
            )
            analisar_e_plotar_comparacao(
                img_base_path="colcheia/base.png", 
                img_comp_path="colcheia/teste3.png"
            )
            analisar_e_plotar_comparacao(
                img_base_path="colcheia/base.png", 
                img_comp_path="colcheia/teste4.png"
            )
        elif escolha == '2':
            pasta_de_amostras = "colcheia"
            extensoes_validas = ('.png', '.jpg', '.jpeg')

            imagens_colcheia = [
                os.path.join(pasta_de_amostras, arquivo) 
                for arquivo in os.listdir(pasta_de_amostras) 
                if arquivo.lower().endswith(extensoes_validas)
            ]

            for img_path in imagens_colcheia:
                vm.mostrar_hue_e_mascara(img_path)

        elif escolha == '0':
            print("Saindo do programa.")
            break
        else:
            print("\nEscolha inválida. Por favor, tente novamente.")

if __name__ == "__main__":
    menu()