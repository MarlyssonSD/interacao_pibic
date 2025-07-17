import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import view_masks as vm
import dominant_colors_hsv as dch

def analisar_e_plotar_comparacao(img_base_path, img_comp_path, threshold=0.6):
    img_base = cv2.imread(img_base_path)
    img_comp = cv2.imread(img_comp_path)

    if img_base is None or img_comp is None:
        print(f"ERRO: Não foi possível carregar as imagens.")
        return

    # Converte para HSV
    hsv_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2HSV)
    hsv_comp = cv2.cvtColor(img_comp, cv2.COLOR_BGR2HSV)

    # Máscaras para vermelho em hsv
    # lower_red1 = np.array([0, 20, 20])
    # upper_red1 = np.array([15, 255, 255])
    # lower_red2 = np.array([165, 20, 20])
    # upper_red2 = np.array([179, 255, 255])
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([165, 70, 70])
    upper_red2 = np.array([179, 255, 255])

    mask_base = cv2.bitwise_or(cv2.inRange(hsv_base, lower_red1, upper_red1), cv2.inRange(hsv_base, lower_red2, upper_red2))
    mask_comp = cv2.bitwise_or(cv2.inRange(hsv_comp, lower_red1, upper_red1), cv2.inRange(hsv_comp, lower_red2, upper_red2))

    # Calculo e normalização dos histogramas
    hist_base = cv2.calcHist([hsv_base], [0], mask_base, [180], [0, 180])
    hist_comp = cv2.calcHist([hsv_comp], [0], mask_comp, [180], [0, 180])
    # hist_base = cv2.calcHist([hsv_base], [0], None, [180], [0, 180])
    # hist_comp = cv2.calcHist([hsv_comp], [0], None, [180], [0, 180])

    cv2.normalize(hist_base, hist_base, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_comp, hist_comp, 0, 1, cv2.NORM_MINMAX)

    # Comparação de histogramas
    score = cv2.compareHist(hist_base, hist_comp, cv2.HISTCMP_CORREL)
    interacao = score < threshold
    
    print(f"Analisando '{img_base_path}' vs '{img_comp_path}'...")
    print(f"Score de Correlação Calculado: {score:.4f}")
    print(f"O score < {threshold}? {'Sim' if interacao else 'Não'}")
    print(f"Resultado Final: {'Interação com alvo detectada' if interacao else 'Não houve interação com o alvo'}")

    # Plotar os resultados
    img_base_rgb = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
    img_comp_rgb = cv2.cvtColor(img_comp, cv2.COLOR_BGR2RGB)
    
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    resultado_texto = f"Resultado: {'Interação com alvo detectada' if interacao else 'Não houve interação com o alvo'}"
    score_texto = f"Score de Correlação: {score:.4f}"
    fig.suptitle(f"{resultado_texto} | {score_texto}", fontsize=16, weight='bold')

    axs[0, 0].imshow(img_base_rgb); axs[0, 0].set_title("Recorte Base"); axs[0, 0].axis('off')
    axs[0, 1].plot(hist_base, color='red'); axs[0, 1].set_title("Histograma de Matiz (Hue) - Base"); axs[0, 1].set_xlim([0, 180]); axs[0, 1].grid(True, alpha=0.5)
    axs[1, 0].imshow(img_comp_rgb); axs[1, 0].set_title("Recorte de Comparação"); axs[1, 0].axis('off')
    axs[1, 1].plot(hist_comp, color='blue'); axs[1, 1].set_title("Histograma de Matiz (Hue) - Comparação"); axs[1, 1].set_xlim([0, 180]); axs[1, 1].grid(True, alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
        
    # salvar o gráfico em uma pasta
    nome_base = os.path.splitext(os.path.basename(img_comp_path))[0]
    output_dir = "comparacoes_salvas_correl"
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