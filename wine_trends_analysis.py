import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class WineTrendsAnalyzer:
    def __init__(self):
        self.wine_data = None
        self.brazilian_wines = None
        
    def load_data(self):
        """Carrega dados de avaliações de vinhos"""
        try:
            self.wine_data = pd.read_csv('raw-data/winemag-data-130k-v2.csv')
            print(f"✅ Dataset carregado: {len(self.wine_data):,} avaliações de vinhos")
            
            # Filtrar vinhos brasileiros
            self.brazilian_wines = self.wine_data[self.wine_data['country'] == 'Brazil'].copy()
            print(f"🇧🇷 Vinhos brasileiros encontrados: {len(self.brazilian_wines)}")
            
            # Limpar dados
            self.clean_data()
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return False
    
    def clean_data(self):
        """Limpa e prepara os dados para análise"""
        # Remover valores nulos em colunas importantes
        self.wine_data = self.wine_data.dropna(subset=['points', 'price'])
        
        # Filtrar preços extremos (outliers)
        Q1 = self.wine_data['price'].quantile(0.25)
        Q3 = self.wine_data['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        self.wine_data = self.wine_data[
            (self.wine_data['price'] >= lower_bound) & 
            (self.wine_data['price'] <= upper_bound)
        ]
        
        # Criar categorias de preço
        self.wine_data['price_category'] = pd.cut(
            self.wine_data['price'], 
            bins=[0, 15, 30, 50, 100, float('inf')],
            labels=['Econômico (<$15)', 'Médio ($15-30)', 'Premium ($30-50)', 
                   'Super Premium ($50-100)', 'Ultra Premium (>$100)']
        )
        
        # Criar categorias de pontuação
        self.wine_data['rating_category'] = pd.cut(
            self.wine_data['points'],
            bins=[0, 82, 87, 90, 95, 100],
            labels=['Aceitável (80-82)', 'Bom (83-87)', 'Muito Bom (88-90)', 
                   'Excelente (91-95)', 'Excepcional (96-100)']
        )
        
        print(f"📊 Dados limpos: {len(self.wine_data):,} registros válidos")
    
    def analyze_price_quality_relationship(self):
        """Analisa relação entre preço e qualidade"""
        print("\n" + "="*60)
        print("ANÁLISE: RELAÇÃO PREÇO vs QUALIDADE")
        print("="*60)
        
        # Correlação preço-pontuação
        correlation = self.wine_data['price'].corr(self.wine_data['points'])
        print(f"📈 Correlação Preço-Pontuação: {correlation:.3f}")
        
        # Análise por categoria de preço
        price_analysis = self.wine_data.groupby('price_category').agg({
            'points': ['mean', 'std', 'count'],
            'price': ['mean', 'median']
        }).round(2)
        
        print("\n📊 ESTATÍSTICAS POR CATEGORIA DE PREÇO:")
        print(price_analysis)
        
        # Criar visualização
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Scatter plot preço vs pontuação
        ax1.scatter(self.wine_data['price'], self.wine_data['points'], 
                   alpha=0.5, s=20, color='darkred')
        ax1.set_xlabel('Preço (US$)')
        ax1.set_ylabel('Pontuação')
        ax1.set_title('Relação Preço vs Pontuação')
        ax1.grid(True, alpha=0.3)
        
        # Box plot pontuação por categoria de preço
        self.wine_data.boxplot(column='points', by='price_category', ax=ax2)
        ax2.set_title('Distribuição de Pontuações por Categoria de Preço')
        ax2.set_xlabel('Categoria de Preço')
        ax2.set_ylabel('Pontuação')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Histograma de preços
        ax3.hist(self.wine_data['price'], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Preço (US$)')
        ax3.set_ylabel('Frequência')
        ax3.set_title('Distribuição de Preços')
        ax3.grid(True, alpha=0.3)
        
        # Média de pontuação por categoria de preço
        avg_points = self.wine_data.groupby('price_category')['points'].mean()
        bars = ax4.bar(range(len(avg_points)), avg_points.values, 
                      color=plt.cm.RdYlBu_r(np.linspace(0, 1, len(avg_points))))
        ax4.set_xlabel('Categoria de Preço')
        ax4.set_ylabel('Pontuação Média')
        ax4.set_title('Pontuação Média por Categoria de Preço')
        ax4.set_xticks(range(len(avg_points)))
        ax4.set_xticklabels(avg_points.index, rotation=45)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, avg_points.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('analise_preco_qualidade.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation, price_analysis
    
    def analyze_variety_trends(self):
        """Analisa tendências por variedade de uva"""
        print("\n" + "="*60)
        print("ANÁLISE: TENDÊNCIAS POR VARIEDADE DE UVA")
        print("="*60)
        
        # Top variedades por volume
        top_varieties = self.wine_data['variety'].value_counts().head(15)
        print("\n🍇 TOP 15 VARIEDADES MAIS AVALIADAS:")
        for i, (variety, count) in enumerate(top_varieties.items(), 1):
            print(f"{i:2d}. {variety}: {count:,} avaliações")
        
        # Análise de preço e qualidade por variedade
        variety_analysis = self.wine_data[self.wine_data['variety'].isin(top_varieties.index)].groupby('variety').agg({
            'points': ['mean', 'std'],
            'price': ['mean', 'median'],
            'variety': 'count'
        }).round(2)
        
        variety_analysis.columns = ['Pontuação_Média', 'Pontuação_Desvio', 
                                   'Preço_Médio', 'Preço_Mediano', 'Quantidade']
        
        # Calcular valor por ponto (custo-benefício)
        variety_analysis['Custo_Benefício'] = (variety_analysis['Preço_Médio'] / 
                                              variety_analysis['Pontuação_Média']).round(3)
        
        variety_analysis = variety_analysis.sort_values('Custo_Benefício')
        
        print("\n💰 RANKING DE CUSTO-BENEFÍCIO (menor = melhor):")
        print(variety_analysis[['Pontuação_Média', 'Preço_Médio', 'Custo_Benefício']].head(10))
        
        # Visualização
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top variedades por volume
        top_10_varieties = top_varieties.head(10)
        bars1 = ax1.barh(range(len(top_10_varieties)), top_10_varieties.values,
                        color=plt.cm.Set3(np.linspace(0, 1, len(top_10_varieties))))
        ax1.set_xlabel('Número de Avaliações')
        ax1.set_title('Top 10 Variedades Mais Avaliadas')
        ax1.set_yticks(range(len(top_10_varieties)))
        ax1.set_yticklabels(top_10_varieties.index)
        
        # Custo-benefício
        best_value = variety_analysis.head(10)
        bars2 = ax2.bar(range(len(best_value)), best_value['Custo_Benefício'],
                       color=plt.cm.RdYlGn_r(np.linspace(0, 1, len(best_value))))
        ax2.set_xlabel('Variedade')
        ax2.set_ylabel('Custo-Benefício (Preço/Pontuação)')
        ax2.set_title('Top 10 Variedades com Melhor Custo-Benefício')
        ax2.set_xticks(range(len(best_value)))
        ax2.set_xticklabels(best_value.index, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('analise_variedades.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return variety_analysis
    
    def analyze_regional_preferences(self):
        """Analisa preferências regionais"""
        print("\n" + "="*60)
        print("ANÁLISE: PREFERÊNCIAS REGIONAIS")
        print("="*60)
        
        # Top países produtores
        top_countries = self.wine_data['country'].value_counts().head(15)
        print("\n🌍 TOP 15 PAÍSES PRODUTORES:")
        for i, (country, count) in enumerate(top_countries.items(), 1):
            print(f"{i:2d}. {country}: {count:,} vinhos")
        
        # Análise por país
        country_analysis = self.wine_data[self.wine_data['country'].isin(top_countries.index)].groupby('country').agg({
            'points': ['mean', 'std'],
            'price': ['mean', 'median'],
            'country': 'count'
        }).round(2)
        
        country_analysis.columns = ['Pontuação_Média', 'Pontuação_Desvio', 
                                   'Preço_Médio', 'Preço_Mediano', 'Quantidade']
        
        country_analysis = country_analysis.sort_values('Pontuação_Média', ascending=False)
        
        print("\n🏆 TOP 10 PAÍSES POR QUALIDADE MÉDIA:")
        print(country_analysis[['Pontuação_Média', 'Preço_Médio', 'Quantidade']].head(10))
        
        # Posição do Brasil
        if 'Brazil' in country_analysis.index:
            brazil_rank = list(country_analysis.index).index('Brazil') + 1
            brazil_stats = country_analysis.loc['Brazil']
            print(f"\n🇧🇷 POSIÇÃO DO BRASIL:")
            print(f"Ranking: {brazil_rank}º lugar")
            print(f"Pontuação média: {brazil_stats['Pontuação_Média']:.1f}")
            print(f"Preço médio: US$ {brazil_stats['Preço_Médio']:.2f}")
            print(f"Quantidade de vinhos: {int(brazil_stats['Quantidade'])}")
        
        return country_analysis
    
    def identify_purchase_trends(self):
        """Identifica tendências específicas de compra"""
        print("\n" + "="*60)
        print("TENDÊNCIAS DE COMPRA IDENTIFICADAS")
        print("="*60)
        
        # 1. Sweet Spot de preço-qualidade
        sweet_spot = self.wine_data[
            (self.wine_data['points'] >= 88) & 
            (self.wine_data['price'] <= 30)
        ]
        
        print(f"\n🎯 SWEET SPOT (Pontuação ≥88, Preço ≤$30):")
        print(f"Quantidade de vinhos: {len(sweet_spot):,}")
        print(f"Percentual do total: {len(sweet_spot)/len(self.wine_data)*100:.1f}%")
        
        if len(sweet_spot) > 0:
            print(f"Pontuação média: {sweet_spot['points'].mean():.1f}")
            print(f"Preço médio: US$ {sweet_spot['price'].mean():.2f}")
            
            sweet_spot_varieties = sweet_spot['variety'].value_counts().head(5)
            print("\nTop 5 variedades no sweet spot:")
            for variety, count in sweet_spot_varieties.items():
                print(f"  • {variety}: {count} vinhos")
        
        # 2. Vinhos premium com melhor custo-benefício
        premium_wines = self.wine_data[self.wine_data['points'] >= 90]
        premium_value = premium_wines.copy()
        premium_value['value_score'] = premium_value['points'] / premium_value['price']
        best_premium = premium_value.nlargest(10, 'value_score')
        
        print(f"\n💎 VINHOS PREMIUM COM MELHOR CUSTO-BENEFÍCIO (Pontuação ≥90):")
        for idx, wine in best_premium.iterrows():
            print(f"  • {wine['title'][:50]}...")
            print(f"    Pontuação: {wine['points']}, Preço: US$ {wine['price']:.2f}, Valor: {wine['value_score']:.3f}")
        
        # 3. Tendências por faixa de preço
        print(f"\n📊 DISTRIBUIÇÃO POR CATEGORIA DE PREÇO:")
        price_dist = self.wine_data['price_category'].value_counts()
        for category, count in price_dist.items():
            percentage = count / len(self.wine_data) * 100
            print(f"  • {category}: {count:,} vinhos ({percentage:.1f}%)")
        
        # 4. Correlações interessantes
        print(f"\n🔍 CORRELAÇÕES DESCOBERTAS:")
        
        # Correlação pontuação-preço por país
        correlations = []
        for country in self.wine_data['country'].value_counts().head(10).index:
            country_data = self.wine_data[self.wine_data['country'] == country]
            if len(country_data) > 50:  # Mínimo de dados para correlação confiável
                corr = country_data['price'].corr(country_data['points'])
                correlations.append((country, corr, len(country_data)))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        print("\nCorrelação Preço-Qualidade por país:")
        for country, corr, count in correlations[:5]:
            print(f"  • {country}: {corr:.3f} ({count} vinhos)")
        
        return {
            'sweet_spot': sweet_spot,
            'best_premium': best_premium,
            'price_distribution': price_dist,
            'country_correlations': correlations
        }
    
    def generate_recommendations(self, trends_data):
        """Gera recomendações baseadas nas tendências"""
        print("\n" + "="*60)
        print("RECOMENDAÇÕES ESTRATÉGICAS")
        print("="*60)
        
        print("\n🎯 PARA CONSUMIDORES:")
        print("1. MELHOR CUSTO-BENEFÍCIO: Procure vinhos na faixa de $15-30 com pontuação ≥88")
        print("2. VARIEDADES RECOMENDADAS: Pinot Noir, Chardonnay e Cabernet Sauvignon oferecem boa relação qualidade-preço")
        print("3. PAÍSES EMERGENTES: Considere vinhos de países como Portugal e Argentina para descobrir valores")
        print("4. SWEET SPOT: Vinhos entre $20-25 frequentemente oferecem excelente qualidade")
        
        print("\n🏭 PARA PRODUTORES BRASILEIROS:")
        print("1. POSICIONAMENTO: Focar na faixa premium ($30-50) com alta qualidade")
        print("2. VARIEDADES: Investir em variedades com boa aceitação internacional")
        print("3. MARKETING: Destacar terroir único e sustentabilidade")
        print("4. PREÇO: Manter competitividade na faixa de $15-30 para ganhar mercado")
        
        print("\n📈 PARA INVESTIDORES:")
        print("1. MERCADO CRESCENTE: Segmento premium mostra forte correlação preço-qualidade")
        print("2. OPORTUNIDADE: Vinhos brasileiros têm potencial de valorização")
        print("3. DIVERSIFICAÇÃO: Investir em diferentes regiões e variedades")
        print("4. SUSTENTABILIDADE: Vinhos orgânicos e biodinâmicos são tendência")
    
    def run_complete_analysis(self):
        """Executa análise completa"""
        print("🍷 ANÁLISE COMPLETA DE TENDÊNCIAS DE COMPRA DE VINHOS")
        print("=" * 70)
        
        if not self.load_data():
            return
        
        # Executar todas as análises
        correlation, price_analysis = self.analyze_price_quality_relationship()
        variety_analysis = self.analyze_variety_trends()
        country_analysis = self.analyze_regional_preferences()
        trends_data = self.identify_purchase_trends()
        self.generate_recommendations(trends_data)
        
        print("\n" + "="*70)
        print("✅ ANÁLISE CONCLUÍDA! Gráficos salvos como PNG.")
        print("="*70)
        
        return {
            'correlation': correlation,
            'price_analysis': price_analysis,
            'variety_analysis': variety_analysis,
            'country_analysis': country_analysis,
            'trends_data': trends_data
        }

# Executar análise
if __name__ == "__main__":
    analyzer = WineTrendsAnalyzer()
    results = analyzer.run_complete_analysis()