import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WineExportAnalyzer:
    def __init__(self):
        self.export_data = None
        self.climate_data = None
        self.demographic_data = None
        self.economic_data = None
        self.wine_ratings = None
        
    def generate_mock_data(self):
        """Gera dados mock para análise"""
        # Países de destino principais
        countries = ['Estados Unidos', 'Reino Unido', 'Alemanha', 'Canadá', 
                    'França', 'Japão', 'China', 'Argentina', 'Paraguai', 'Uruguai']
        
        # Dados de exportação (2009-2023)
        years = list(range(2009, 2024))
        export_data = []
        
        for year in years:
            for country in countries:
                # Simulando crescimento com variações
                base_volume = np.random.normal(1000000, 200000)  # litros
                growth_factor = 1 + (year - 2009) * 0.05 + np.random.normal(0, 0.1)
                volume = max(0, base_volume * growth_factor)
                
                # Preço por litro varia por país e ano
                base_price = np.random.uniform(2.5, 8.0)  # US$ por litro
                price_factor = 1 + (year - 2009) * 0.03
                price_per_liter = base_price * price_factor
                
                export_data.append({
                    'ano': year,
                    'pais_destino': country,
                    'volume_litros': volume,
                    'valor_usd': volume * price_per_liter
                })
        
        self.export_data = pd.DataFrame(export_data)
        
        # Dados climáticos do Brasil
        climate_data = []
        for year in years:
            climate_data.append({
                'ano': year,
                'temperatura_media': np.random.normal(25, 2),
                'precipitacao_mm': np.random.normal(1200, 200),
                'dias_sol': np.random.normal(250, 30)
            })
        
        self.climate_data = pd.DataFrame(climate_data)
        
        # Dados demográficos
        demographic_data = []
        for year in years:
            demographic_data.append({
                'ano': year,
                'populacao_brasil': 200000000 + (year - 2009) * 2000000,
                'renda_per_capita': 8000 + (year - 2009) * 200 + np.random.normal(0, 500)
            })
        
        self.demographic_data = pd.DataFrame(demographic_data)
        
        # Dados econômicos
        economic_data = []
        for year in years:
            economic_data.append({
                'ano': year,
                'pib_brasil': 2000000000000 + (year - 2009) * 50000000000,
                'taxa_cambio_usd_brl': np.random.uniform(1.8, 5.5),
                'inflacao': np.random.uniform(2, 12)
            })
        
        self.economic_data = pd.DataFrame(economic_data)
        
        # Avaliações de vinhos
        wine_ratings = []
        for year in years:
            wine_ratings.append({
                'ano': year,
                'pontuacao_media': np.random.uniform(82, 92),
                'num_avaliacoes': np.random.randint(500, 2000),
                'preco_medio_garrafa': np.random.uniform(15, 45)
            })
        
        self.wine_ratings = pd.DataFrame(wine_ratings)
    
    def create_export_evolution_chart(self):
        """Gráfico de evolução das exportações"""
        yearly_data = self.export_data.groupby('ano').agg({
            'volume_litros': 'sum',
            'valor_usd': 'sum'
        }).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Volume
        ax1.plot(yearly_data['ano'], yearly_data['volume_litros']/1000000, 
                marker='o', linewidth=3, markersize=8, color='#2E8B57')
        ax1.set_title('Evolução do Volume de Exportação de Vinhos Brasileiros', 
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Volume (Milhões de Litros)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Valor
        ax2.plot(yearly_data['ano'], yearly_data['valor_usd']/1000000, 
                marker='s', linewidth=3, markersize=8, color='#8B0000')
        ax2.set_title('Evolução do Valor das Exportações', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Ano', fontsize=12)
        ax2.set_ylabel('Valor (Milhões USD)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evolucao_exportacoes.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_country_analysis(self):
        """Análise por país de destino"""
        country_data = self.export_data.groupby('pais_destino').agg({
            'volume_litros': 'sum',
            'valor_usd': 'sum'
        }).reset_index()
        
        country_data = country_data.sort_values('valor_usd', ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top países por volume
        top_countries_vol = country_data.head(8)
        bars1 = ax1.bar(range(len(top_countries_vol)), 
                        top_countries_vol['volume_litros']/1000000,
                        color=plt.cm.Set3(np.linspace(0, 1, len(top_countries_vol))))
        ax1.set_title('Top Países por Volume (2009-2023)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Volume (Milhões de Litros)', fontsize=12)
        ax1.set_xticks(range(len(top_countries_vol)))
        ax1.set_xticklabels(top_countries_vol['pais_destino'], rotation=45, ha='right')
        
        # Top países por valor
        top_countries_val = country_data.head(8)
        bars2 = ax2.bar(range(len(top_countries_val)), 
                        top_countries_val['valor_usd']/1000000,
                        color=plt.cm.Set2(np.linspace(0, 1, len(top_countries_val))))
        ax2.set_title('Top Países por Valor (2009-2023)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Valor (Milhões USD)', fontsize=12)
        ax2.set_xticks(range(len(top_countries_val)))
        ax2.set_xticklabels(top_countries_val['pais_destino'], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('analise_paises.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_correlation_analysis(self):
        """Análise de correlação com fatores externos"""
        # Consolidar dados por ano
        yearly_export = self.export_data.groupby('ano').agg({
            'volume_litros': 'sum',
            'valor_usd': 'sum'
        }).reset_index()
        
        # Merge com outros dados
        analysis_data = yearly_export.merge(self.climate_data, on='ano')
        analysis_data = analysis_data.merge(self.economic_data, on='ano')
        analysis_data = analysis_data.merge(self.wine_ratings, on='ano')
        
        # Matriz de correlação
        corr_columns = ['volume_litros', 'valor_usd', 'temperatura_media', 
                       'precipitacao_mm', 'taxa_cambio_usd_brl', 'pontuacao_media']
        corr_matrix = analysis_data[corr_columns].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Matriz de Correlação: Exportações vs Fatores Externos', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlacao_fatores.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return analysis_data
    
    def create_future_projections(self, analysis_data):
        """Projeções futuras"""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        # Preparar dados para projeção
        X = analysis_data[['ano']].values
        y_volume = analysis_data['volume_litros'].values
        y_value = analysis_data['valor_usd'].values
        
        # Modelo de regressão polinomial
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        
        model_volume = LinearRegression().fit(X_poly, y_volume)
        model_value = LinearRegression().fit(X_poly, y_value)
        
        # Projeções para 2024-2028
        future_years = np.array([[year] for year in range(2024, 2029)])
        future_years_poly = poly_features.transform(future_years)
        
        projected_volume = model_volume.predict(future_years_poly)
        projected_value = model_value.predict(future_years_poly)
        
        # Visualização
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Volume
        ax1.plot(analysis_data['ano'], analysis_data['volume_litros']/1000000, 
                'o-', label='Dados Históricos', linewidth=2, markersize=6)
        ax1.plot(range(2024, 2029), projected_volume/1000000, 
                's--', label='Projeção', linewidth=2, markersize=6, color='red')
        ax1.set_title('Projeção de Volume de Exportação (2024-2028)', 
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Volume (Milhões de Litros)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Valor
        ax2.plot(analysis_data['ano'], analysis_data['valor_usd']/1000000, 
                'o-', label='Dados Históricos', linewidth=2, markersize=6)
        ax2.plot(range(2024, 2029), projected_value/1000000, 
                's--', label='Projeção', linewidth=2, markersize=6, color='red')
        ax2.set_title('Projeção de Valor das Exportações (2024-2028)', 
                     fontsize=16, fontweight='bold')
        ax2.set_xlabel('Ano', fontsize=12)
        ax2.set_ylabel('Valor (Milhões USD)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('projecoes_futuras.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return projected_volume, projected_value
    
    def generate_executive_summary(self, analysis_data, projected_volume, projected_value):
        """Gera relatório executivo"""
        total_volume_15y = analysis_data['volume_litros'].sum() / 1000000  # Milhões de litros
        total_value_15y = analysis_data['valor_usd'].sum() / 1000000  # Milhões USD
        avg_price_per_liter = total_value_15y / total_volume_15y
        
        growth_rate_volume = ((analysis_data['volume_litros'].iloc[-1] / 
                              analysis_data['volume_litros'].iloc[0]) ** (1/14) - 1) * 100
        growth_rate_value = ((analysis_data['valor_usd'].iloc[-1] / 
                             analysis_data['valor_usd'].iloc[0]) ** (1/14) - 1) * 100
        
        print("=" * 80)
        print("RELATÓRIO EXECUTIVO - EXPORTAÇÕES DE VINHO BRASILEIRO (2009-2023)")
        print("=" * 80)
        print(f"\n📊 RESUMO GERAL:")
        print(f"• Volume total exportado (15 anos): {total_volume_15y:.1f} milhões de litros")
        print(f"• Valor total das exportações: US$ {total_value_15y:.1f} milhões")
        print(f"• Preço médio por litro: US$ {avg_price_per_liter:.2f}")
        print(f"• Taxa de crescimento anual (volume): {growth_rate_volume:.1f}%")
        print(f"• Taxa de crescimento anual (valor): {growth_rate_value:.1f}%")
        
        print(f"\n🎯 PRINCIPAIS MERCADOS:")
        top_markets = self.export_data.groupby('pais_destino')['valor_usd'].sum().sort_values(ascending=False).head(5)
        for i, (country, value) in enumerate(top_markets.items(), 1):
            print(f"{i}. {country}: US$ {value/1000000:.1f} milhões")
        
        print(f"\n🔮 PROJEÇÕES 2024-2028:")
        print(f"• Volume projetado 2028: {projected_volume[-1]/1000000:.1f} milhões de litros")
        print(f"• Valor projetado 2028: US$ {projected_value[-1]/1000000:.1f} milhões")
        
        print(f"\n💡 RECOMENDAÇÕES ESTRATÉGICAS:")
        print("1. Diversificação de mercados: Explorar novos destinos na Ásia e África")
        print("2. Melhoria da qualidade: Investir em tecnologia e certificações")
        print("3. Marketing digital: Fortalecer presença online nos mercados-alvo")
        print("4. Sustentabilidade: Implementar práticas ESG para atrair investidores")
        print("5. Parcerias estratégicas: Alianças com distribuidores internacionais")
        print("=" * 80)

# Executar análise completa
if __name__ == "__main__":
    analyzer = WineExportAnalyzer()
    
    print("Gerando dados mock para análise...")
    analyzer.generate_mock_data()
    
    print("Criando visualizações...")
    analyzer.create_export_evolution_chart()
    analyzer.create_country_analysis()
    
    analysis_data = analyzer.create_correlation_analysis()
    projected_volume, projected_value = analyzer.create_future_projections(analysis_data)
    
    analyzer.generate_executive_summary(analysis_data, projected_volume, projected_value)
    
    print("\nAnálise completa! Gráficos salvos como PNG.")