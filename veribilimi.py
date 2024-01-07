import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("DOSYAKONUMU" ,error_bad_lines=False)

# Rastgele 1-10 arası puanlar oluşturun
np.random.seed(42)
df['average_rating'] = np.round(np.random.uniform(low=1, high=5, size=len(df)))

# print(df.head())

print(df.describe())

#.............................................................................

# 'Calories' sütununu kullanarak en yüksek 10 değeri seç
top_ten = df.sort_values(by='Calories', ascending=False).head(10)

# Çubuk grafiği çiz
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10, 10))
sns.barplot(x="Calories", y="drink", data=top_ten, palette='inferno')

# Grafik başlığı ve ekseni etiketleri
plt.title('En Yüksek 10 Kalori Değerine Sahip İçecekler')
plt.xlabel('Calories')
plt.ylabel('Drink')

# Grafiği göster
plt.show()

#..................................................................

# İçecek türlerine göre ortalama kalori içeriği
avg_calories_by_type = df.groupby('type')['Calories'].mean().sort_values(ascending=False)

# Çubuk grafiğini çizin
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_calories_by_type.values, y=avg_calories_by_type.index, palette='viridis')
plt.title('İçecek Türlerine Göre Ortalama Kalori İçeriği')
plt.xlabel('Ortalama Kalori')
plt.ylabel('İçecek Türü')
plt.show()

#............................................................................

# Kaloriye göre azalan sırada ve ilk 10 içeceği seçme
top_10_calories = df.sort_values(by='Calories', ascending=False).head(10)

# Çubuk grafiğini çizin
plt.figure(figsize=(12, 8))
sns.barplot(x='drink', y='Caffeine (mg)', data=top_10_calories, palette='viridis')
plt.title('En Yüksek 10 Kaloriye Sahip İçeceklere Göre Kafein Miktarı')
plt.xlabel('İçecek Adı')
plt.ylabel('Kafein Miktarı (mg)')
plt.xticks(rotation=45, ha='right')  # İçecek adlarını yatay olarak düzenleme
plt.show()

#.....................................................

#kalori,kafein ve puana göre ilişkili bir plot oluşturma 

# Scatter plot oluşturma
plt.figure(figsize=[15,10])
sns.scatterplot(x='Caffeine (mg)', y='Calories', hue='average_rating', size='average_rating', data=df, palette='viridis', sizes=(50, 500))
plt.title('İlişki: Kafein Miktarı, Kalori ve Puan', fontsize=20)
plt.xlabel('Kafein Miktarı (mg)', fontsize=15)
plt.ylabel('Kalori', fontsize=15)
plt.show()


#.........................................................................................


from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

def CoffeeRecommender(coffee_name, df):
    # Kahve adına karşılık gelen kahve indeksini bulun
    coffee_id = df[df['drink'] == coffee_name].index
    coffee_id = coffee_id[0]

    # MinMaxScaler kullanarak özellikleri ölçeklendirin
    min_max_scaler = MinMaxScaler()
    features_scaled = min_max_scaler.fit_transform(df[['Caffeine (mg)', 'Calories', 'average_rating']])

    # NearestNeighbors modelini oluşturun
    model = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
    model.fit(features_scaled)

    # Modeli kullanarak komşu noktaları ve mesafeleri hesaplayın
    distances, neighbors = model.kneighbors(features_scaled)

    # Kahvenin önerilmesi için komşu kahveleri bulun
    coffee_list_name = []
    for newid in neighbors[coffee_id]:
        coffee_list_name.append(df.loc[newid, 'drink'])

    return coffee_list_name

# Örnek kullanım
recommended_coffees = CoffeeRecommender('Zola Coconut Water Espresso', df)
print(recommended_coffees)