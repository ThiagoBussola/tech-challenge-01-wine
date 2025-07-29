import pandas as pd
import numpy as np
from pathlib import Path

class DataProcessor:
    """Classe para processar os dados reais dos arquivos fornecidos"""
    
    def __init__(self, data_path="raw-data"):
        self.data_path = Path(data_path)
        self.wine_reviews = None
        self.embrapa_data = None
    
    def load_wine_reviews(self):
        """Carrega dados de avaliações de vinhos"""
        try:
            file_path = self.data_path / "winemag-data-130k-v2.csv"
            self.wine_reviews = pd.read_csv(file_path)
            print(f"Dados de avaliações carregados: {len(self.wine_reviews)} registros")
            return self.wine_reviews
        except Exception as e:
            print(f"Erro ao carregar dados de avaliações: {e}")
            return None
    
    def load_embrapa_data(self):
        """Carrega dados da Embrapa"""
        try:
            file_path = self.data_path / "embrapa e OIV(Definitiva 2.0).CSV (2).xlsx"
            self.embrapa_data = pd.read_excel(file_path)
            print(f"Dados da Embrapa carregados: {len(self.embrapa_data)} registros")
            return self.embrapa_data
        except Exception as e:
            print(f"Erro ao carregar dados da Embrapa: {e}")
            return None
    
    def analyze_wine_reviews(self):
        """Analisa dados de avaliações de vinhos"""
        if self.wine_reviews is None:
            self.load_wine_reviews()
        
        if self.wine_reviews is not None:
            # Filtrar vinhos brasileiros
            brazilian_wines = self.wine_reviews[self.wine_reviews['country'] == 'Brazil']
            
            analysis = {
                'total_reviews': len(brazilian_wines),
                'avg_points': brazilian_wines['points'].mean(),
                'avg_price': brazilian_wines['price'].mean(),
                'price_range': (brazilian_wines['price'].min(), brazilian_wines['price'].max()),
                'top_varieties': brazilian_wines['variety'].value_counts().head(10)
            }
            
            return analysis
        return None
    
    def explore_datasets(self):
        """Explora a estrutura dos datasets"""
        print("=" * 60)
        print("EXPLORAÇÃO DOS DATASETS")
        print("=" * 60)
        
        # Carregar e explorar dados de avaliações
        print("\n📊 DATASET DE AVALIAÇÕES DE VINHOS:")
        wine_data = self.load_wine_reviews()
        if wine_data is not None:
            print(f"Colunas disponíveis: {list(wine_data.columns)}")
            print(f"Países únicos: {wine_data['country'].nunique()}")
            print(f"Vinhos brasileiros: {len(wine_data[wine_data['country'] == 'Brazil'])}")
            
            # Análise dos vinhos brasileiros
            brazilian_analysis = self.analyze_wine_reviews()
            if brazilian_analysis:
                print("\n🇧🇷 ANÁLISE DOS VINHOS BRASILEIROS:")
                print(f"Total de avaliações: {brazilian_analysis['total_reviews']}")
                print(f"Pontuação média: {brazilian_analysis['avg_points']:.1f}")
                print(f"Preço médio: US$ {brazilian_analysis['avg_price']:.2f}")
                print(f"Faixa de preços: US$ {brazilian_analysis['price_range'][0]:.2f} - US$ {brazilian_analysis['price_range'][1]:.2f}")
                print("\nTop 5 variedades:")
                for variety, count in brazilian_analysis['top_varieties'].head(5).items():
                    print(f"  • {variety}: {count} avaliações")
        
        # Carregar e explorar dados da Embrapa
        print("\n📈 DATASET DA EMBRAPA:")
        embrapa_data = self.load_embrapa_data()
        if embrapa_data is not None:
            print(f"Colunas disponíveis: {list(embrapa_data.columns)}")
            print(f"Número de registros: {len(embrapa_data)}")
            print(f"Primeiras linhas:")
            print(embrapa_data.head())
        
        print("\n" + "=" * 60)
        print("EXPLORAÇÃO CONCLUÍDA")
        print("=" * 60)

# Código de execução principal
if __name__ == "__main__":
    print("🍷 PROCESSADOR DE DADOS - TECH CHALLENGE VINÍCOLA")
    print("Iniciando análise dos datasets...\n")
    
    # Criar instância do processador
    processor = DataProcessor()
    
    # Explorar os datasets
    processor.explore_datasets()
    
    print("\n✅ Análise concluída! Use esta classe em outros scripts para processar os dados.")