import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict
import seaborn as sns
from collections import Counter

thresholds_correl = 0.5
thresholds_coss = 0.4

# Configurar estilo dos gráficos
plt.style.use('default')
sns.set_palette("husl")

def dissimilaridade_cos(hist1: np.ndarray, hist2: np.ndarray) -> float:
    hist1 = hist1.flatten()
    hist2 = hist2.flatten()
    
    # Evitar divisão por zero
    if np.linalg.norm(hist1) == 0 or np.linalg.norm(hist2) == 0:
        return 1.0
    
    # Calcular similaridade de cosseno
    cos_sim = np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))
    
    # Converter para dissimilaridade (1 - similaridade)
    return 1 - cos_sim

def extrair_histograma(imagem: np.ndarray, usar_mascara: bool = False) -> np.ndarray:
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    
    if usar_mascara:
        # Criar máscaras para as duas faixas de vermelho em HSV
        lower_red1 = np.array([0, 70, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([165, 70, 70])
        upper_red2 = np.array([179, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        cv2.bitwise_or(mask1, mask2)
        
        # Calcular histogramas para as duas faixas de vermelho
        hist1 = cv2.calcHist([hsv], [0], mask1, [20], [0, 20])
        hist2 = cv2.calcHist([hsv], [0], mask2, [20], [160, 180])
        
        # Combinar os histogramas
        hist_completo = np.concatenate([hist1, hist2])
    else:
        # Calcular histograma completo sem máscara
        hist_completo = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    
    # Normalizar
    cv2.normalize(hist_completo, hist_completo, 0, 1, cv2.NORM_MINMAX)
    return hist_completo

def analisar_imagem_par(img_base_path: str, img_comp_path: str) -> Dict:
    img_base = cv2.imread(img_base_path)
    img_comp = cv2.imread(img_comp_path)
    
    if img_base is None or img_comp is None:
        raise FileNotFoundError(f"Erro ao carregar imagens: {img_base_path} ou {img_comp_path}")
    
    # Extrair histogramas
    hist_base = extrair_histograma(img_base)
    hist_comp = extrair_histograma(img_comp)
    
    # Calcular métricas
    correlacao = cv2.compareHist(hist_base, hist_comp, cv2.HISTCMP_CORREL)
    dissim_cosseno = dissimilaridade_cos(hist_base, hist_comp)
    
    # Determinar interações 
    interacao_correl = correlacao < thresholds_correl  # Baixa correlação indica interação (imagem modificada)
    interacao_cosseno = dissim_cosseno > thresholds_coss  # Alta dissimilaridade indica interação (maior diferença)
    
    return {
        'img_base_path': img_base_path,
        'img_comp_path': img_comp_path,
        'img_base': img_base,
        'img_comp': img_comp,
        'hist_base': hist_base,
        'hist_comp': hist_comp,
        'correlacao': correlacao,
        'dissim_cosseno': dissim_cosseno,
        'interacao_correl': interacao_correl,
        'interacao_cosseno': interacao_cosseno
    }

def plotar_comparacao_detalhada(resultado: Dict, salvar: bool = True):
    fig = plt.figure(figsize=(16, 12))
    
    # Converter imagens para RGB
    img_base_rgb = cv2.cvtColor(resultado['img_base'], cv2.COLOR_BGR2RGB)
    img_comp_rgb = cv2.cvtColor(resultado['img_comp'], cv2.COLOR_BGR2RGB)
    
    # Layout dos subplots
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)
    
    # Imagens originais
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_base_rgb)
    ax1.set_title('Imagem Base', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_comp_rgb)
    ax2.set_title('Imagem Comparação', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Histogramas
    ax3 = fig.add_subplot(gs[0, 2:])
    ax3.plot(resultado['hist_base'], 'r-', linewidth=2, label='Base', alpha=0.7)
    ax3.plot(resultado['hist_comp'], 'b-', linewidth=2, label='Comparação', alpha=0.7)
    ax3.set_title('Histogramas de Matiz', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Bins do Histograma')
    ax3.set_ylabel('Frequência Normalizada')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Métricas
    ax4 = fig.add_subplot(gs[1, :2])
    metricas = ['Correlação', 'Dissim. Cosseno']
    valores = [resultado['correlacao'], resultado['dissim_cosseno']]
    cores = ['green' if resultado['interacao_correl'] else 'red',
             'green' if resultado['interacao_cosseno'] else 'red']
    
    bars = ax4.bar(metricas, valores, color=cores, alpha=0.7, edgecolor='black')
    ax4.set_title('Métricas de Similaridade', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Valor da Métrica')
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars, valores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Resultado final
    ax5 = fig.add_subplot(gs[1, 2:])
    resultados = ['Correlação', 'Cosseno']
    interacoes = [resultado['interacao_correl'], resultado['interacao_cosseno']]
    cores_resultado = ['green' if inter else 'red' for inter in interacoes]
    labels_resultado = ['Interação' if inter else 'Sem Interação' for inter in interacoes]
    
    bars2 = ax5.bar(resultados, [1, 1], color=cores_resultado, alpha=0.7, edgecolor='black')
    ax5.set_title('Detecção de Interação', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Resultado')
    ax5.set_ylim(0, 1.0)
    
    # Adicionar labels nos resultados
    for i, (bar, label) in enumerate(zip(bars2, labels_resultado)):
        ax5.text(bar.get_x() + bar.get_width()/2., 0.5,
                label, ha='center', va='center', fontweight='bold', 
                color='white' if label == 'Interação' else 'white')
    
    # Resumo textual
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    nome_base = os.path.basename(resultado['img_base_path'])
    nome_comp = os.path.basename(resultado['img_comp_path'])
    
    texto_resumo = f"""
    ANÁLISE COMPARATIVA: {nome_base} vs {nome_comp}
    
    CORRELAÇÃO: {resultado['correlacao']:.4f} -> {f"INTERAÇÃO DETECTADA (< {thresholds_correl})" if resultado['interacao_correl'] else f"SEM INTERAÇÃO (≥ {thresholds_correl})"}
    DISSIMILARIDADE COSSENO: {resultado['dissim_cosseno']:.4f} -> {f"INTERAÇÃO DETECTADA (> {thresholds_coss})" if resultado['interacao_cosseno'] else f"SEM INTERAÇÃO (≤ {thresholds_coss})"}
    
    CONSENSO: {'AMBOS DETECTARAM INTERAÇÃO' if resultado['interacao_correl'] and resultado['interacao_cosseno'] 
              else 'RESULTADOS DIVERGENTES' if resultado['interacao_correl'] != resultado['interacao_cosseno']
              else 'AMBOS NÃO DETECTARAM INTERAÇÃO'}
    """
    
    ax6.text(0.5, 0.5, texto_resumo, transform=ax6.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle(f'Análise Detalhada de Interação com Alvo', fontsize=16, fontweight='bold')
    
    if salvar:
        os.makedirs('analises_detalhadas', exist_ok=True)
        nome_arquivo = f"analise_{os.path.splitext(nome_comp)[0]}.png"
        caminho_salvar = os.path.join('analises_detalhadas', nome_arquivo)
        plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight')
        print(f"Análise detalhada salva em: {caminho_salvar}")
    
    # plt.show()

def gerar_relatorio_comparativo(resultados: List[Dict], salvar: bool = True):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))  # Tamanho maior da figura
    
    # Extrair dados
    nomes = [os.path.splitext(os.path.basename(r['img_comp_path']))[0] for r in resultados]
    correlacoes = [r['correlacao'] for r in resultados]
    dissim_cossenos = [r['dissim_cosseno'] for r in resultados]
    
    # Gráfico 1: Correlações
    axes[0, 0].bar(range(len(nomes)), correlacoes, 
                   color=['green' if r['interacao_correl'] else 'red' for r in resultados],
                   alpha=0.7, edgecolor='black')
    axes[0, 0].axhline(y=thresholds_correl, color='orange', linestyle='--', alpha=0.8, label=f"Threshold ({thresholds_correl})")
    axes[0, 0].set_title(f"Scores de Correlação (< {thresholds_correl} = Interação)", fontweight='bold')
    axes[0, 0].set_ylabel('Correlação')
    axes[0, 0].set_xticks(range(len(nomes)))
    axes[0, 0].set_xticklabels(nomes, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Dissimilaridades
    axes[0, 1].bar(range(len(nomes)), dissim_cossenos,
                   color=['green' if r['interacao_cosseno'] else 'red' for r in resultados],
                   alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(y=thresholds_coss, color='orange', linestyle='--', alpha=0.8, label=f'Threshold ({thresholds_coss})')
    axes[0, 1].set_title(f"Scores de Dissimilaridade Cosseno (> {thresholds_coss} = Interação)", fontweight='bold')
    axes[0, 1].set_ylabel('Dissimilaridade')
    axes[0, 1].set_xticks(range(len(nomes)))
    axes[0, 1].set_xticklabels(nomes, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Comparação direta
    x_pos = np.arange(len(nomes))
    width = 0.35
    corr_deteccoes = [1 if r['interacao_correl'] else 0 for r in resultados]
    coss_deteccoes = [1 if r['interacao_cosseno'] else 0 for r in resultados]
    
    axes[1, 0].bar(x_pos - width/2, corr_deteccoes, width, label='Correlação', alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, coss_deteccoes, width, label='Cosseno', alpha=0.7)
    axes[1, 0].set_title('Detecções de Interação por Método', fontweight='bold')
    axes[1, 0].set_ylabel('Interação Detectada (1=Sim, 0=Não)')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(nomes, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Matriz de concordância
    concordancias = []
    for r in resultados:
        if r['interacao_correl'] and r['interacao_cosseno']:
            concordancias.append('Ambos Detectaram')
        elif r['interacao_correl'] or r['interacao_cosseno']:
            concordancias.append('Divergência')
        else:
            concordancias.append('Ambos Rejeitaram')
    
    contador = Counter(concordancias)
    
    axes[1, 1].pie(contador.values(), labels=contador.keys(), autopct='%1.1f%%',
                   colors=['green', 'orange', 'red'], startangle=90)
    axes[1, 1].set_title('Concordância entre Métodos', fontweight='bold')
    axes[1, 1].axis('equal')  # Deixa o gráfico de pizza circular
    
    # Título geral e layout ajustado
    plt.suptitle('Relatório Comparativo de Métodos de Detecção', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajuste de layout
    
    # Salvamento
    if salvar:
        os.makedirs('relatorios', exist_ok=True)
        caminho_relatorio = 'relatorios/relatorio_comparativo.png'
        plt.savefig(caminho_relatorio, dpi=300, bbox_inches='tight')
        print(f"Relatório comparativo salvo em: {caminho_relatorio}")
    
def executar_analise_completa():
    # Definir caminhos de teste
    caminhos_teste = [
        "base/colcheia.png",
        "sem-interacao/colcheia/teste1_colcheias.png",
        "sem-interacao/colcheia/teste2_colcheias.png",
        "com-interacao/colcheia/mao_na_frente.png",
        "com-interacao/colcheia/mao_na_frente1.png",
        "com-interacao/colcheia/obstrucao_completa.png",
    ]
    
    img_base = "base/colcheia.png"
    resultados = []
    
    print("Iniciando análise completa...")
    print("=" * 50)
    
    for img_comp in caminhos_teste:
        try:
            print(f"\nAnalisando: {os.path.basename(img_comp)}")
            resultado = analisar_imagem_par(img_base, img_comp)
            resultados.append(resultado)
            
            # Mostrar resultado resumido
            print(f"   Correlação: {resultado['correlacao']:.4f} ({'Interação' if resultado['interacao_correl'] else 'Sem interação'})")
            print(f"   Dissim. Cosseno: {resultado['dissim_cosseno']:.4f} ({'Interação' if resultado['interacao_cosseno'] else 'Sem interação'})")
            
            # Gerar gráfico detalhado
            plotar_comparacao_detalhada(resultado)
            
        except FileNotFoundError as e:
            print(f"Erro: {e}")
    
    if resultados:
        print(f"\nGerando relatório comparativo com {len(resultados)} análises...")
        gerar_relatorio_comparativo(resultados)
        
        # Estatísticas finais
        total_correl = sum(1 for r in resultados if r['interacao_correl'])
        total_cosseno = sum(1 for r in resultados if r['interacao_cosseno'])
        
        print("\n" + "=" * 50)
        print("RESUMO FINAL:")
        print(f"   Total de imagens analisadas: {len(resultados)}")
        print(f"   Interações detectadas (Correlação): {total_correl}/{len(resultados)}")
        print(f"   Interações detectadas (Cosseno): {total_cosseno}/{len(resultados)}")
        print("=" * 50)

def menu():
    """
    Menu principal com opções melhoradas.
    """
    while True:
        print("\n" + "="*60)
        print(" SISTEMA DE ANÁLISE DE INTERAÇÃO COM ALVO")
        print("="*60)
        print("1.  Análise Completa (todos os métodos)")
        print("2.  Análise Individual (escolher imagens)")
        print("3.  Apenas Relatório Comparativo")
        print("4.  Visualização de Máscaras")
        print("0.  Sair")
        print("="*60)
        
        escolha = input("Digite sua opção: ").strip()
        
        if escolha == '1':
            executar_analise_completa()
            
        elif escolha == '2':
            img_base = input("Caminho da imagem base: ").strip()
            img_comp = input("Caminho da imagem de comparação: ").strip()
            
            try:
                resultado = analisar_imagem_par(img_base, img_comp)
                plotar_comparacao_detalhada(resultado)
            except Exception as e:
                print(f" Erro: {e}")
                
        elif escolha == '3':
            # Carregar resultados salvos ou executar análise rápida
            print("Esta opção requer execução prévia da análise completa.")
            
        elif escolha == '4':
            print("Funcionalidade de visualização de máscaras (requer módulo view_masks)")
            
        elif escolha == '0':
            print(" Saindo do programa.")
            break
            
        else:
            print(" Opção inválida. Tente novamente.")

if __name__ == "__main__":
    menu()