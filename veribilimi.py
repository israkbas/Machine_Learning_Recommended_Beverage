import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors



df = pd.read_csv("C:/Users/90544/desktop/caffeine.csv",error_bad_lines=False)

# Rastgele 1-10 arası puanlar oluşturun
np.random.seed(42)
df['average_rating'] = np.random.randint(0,5, size=len(df))

print(df.to_string())

#Veri Temizleme
print(df.head())


print(df.duplicated())

print(df.describe())

#Özellikler arasındaki korelasyon ilişkilerinin incelenmesi
print(df.corr())
df.plot()
plt.show()

#.................................................................
#İçecek dağılımına göre pie plot
drink_types = df['type'].value_counts()

plt.figure(figsize=(10, 6))
plt.pie(drink_types, labels=drink_types.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
plt.legend(title= "İçecek Türleri", loc="upper left")
plt.title('İçecek Türlerine Göre Dağılım')
plt.show()

#.............................................................................

#BoxPlot

plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title('Box Plot: Volume, Calories, Caffeine, average_rating')
plt.show()

#............................................................................

# 'Calories' sütununu kullanarak en yüksek 10 değeri seç
top_ten = df.sort_values(by='Calories', ascending=False).head(10)

# Barplot çiz
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10, 6))
sns.barplot(x="Calories", y="drink", data=top_ten, palette='inferno')
plt.title('En Yüksek 10 Kalori Değerine Sahip İçecekler')
plt.xlabel('Calories')
plt.ylabel('Drink')
plt.tight_layout()
plt.show()

#...........
#histogram
#Ortalama Kalori dağılımı

plt.figure(figsize=(10, 6))
sns.histplot(df['Calories'], bins=20, color='skyblue', kde=True)  
plt.title('Kalori Dağılımı')
plt.xlabel('Kalori')
plt.ylabel('Frekans')
plt.show()

#..................................................................

# İçecek türlerine göre ortalama kalori içeriği
avg_calories_by_type = df.groupby('type')['Calories'].mean().sort_values(ascending=False)

# Barplot
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_calories_by_type.values, y=avg_calories_by_type.index, palette='viridis')
plt.title('İçecek Türlerine Göre Ortalama Kalori İçeriği')
plt.xlabel('Ortalama Kalori')
plt.ylabel('İçecek Türü')
plt.show()


#............................................................................

# Kaloriye göre azalan sırada ve ilk 10 içeceği seçme
top_10_calories = df.sort_values(by='Calories', ascending=False).head(10)

# BarPlot
plt.figure(figsize=(10, 6))
sns.barplot(x='drink', y='Caffeine (mg)',  data=top_10_calories, palette='viridis')
plt.title('En Yüksek 10 Kaloriye Sahip İçeceklere Göre Kafein Miktarı')
plt.xlabel('İçecek Adı')
plt.ylabel('Kafein Miktarı (mg)')
plt.xticks(rotation=45, ha='right')  # İçecek adlarını yatay olarak düzenleme
plt.tight_layout()
plt.show()

#Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='drink', y='Caffeine (mg)', data=top_10_calories, hue='Calories', palette='viridis', s=100)
plt.title('En Yüksek 10 Kaloriye Sahip İçeceklere Göre Kafein Miktarı ve Kalori İlişkisi')
plt.xlabel('İçecek Adı')
plt.ylabel('Kafein Miktarı (mg)')
plt.xticks(rotation=45, ha='right')  # İçecek adlarını yatay olarak düzenleme
plt.legend(title='Kalori', title_fontsize='12')
plt.tight_layout()
plt.show()

#............................................................................
#Pair Plot

#Sayısal Sütunlar arasındaki ilişki
numeric_columns = df.select_dtypes(include='number')
sns.pairplot(numeric_columns)
plt.suptitle('Pair Plot: Sayısal Sütunlar', y=1.02)
plt.show()

#........................................

# Kalori,Kafein ve Puan Arasındaki İlişki
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Caffeine (mg)', y='Calories', hue='average_rating', size='average_rating', data=df, palette='twilight_shifted', sizes=(50, 500))

plt.title('İlişki: Kafein Miktarı, Kalori ve Puan', fontsize=20)
plt.xlabel('Kafein Miktarı (mg)', fontsize=15)
plt.ylabel('Kalori', fontsize=15)
plt.show()

#.............................................................

# Heatmap oluşturma
plt.figure(figsize=(10, 6))

# Veri setindeki sütunları seçme
heatmap_data = df[['Volume (ml)', 'Calories', 'Caffeine (mg)', 'average_rating']]

# Heatmap çizme
sns.heatmap(heatmap_data.corr(), annot=True, fmt='.2f', linewidths=.5)
plt.title('İçecek Verileri Heatmap')
plt.show()

#.........................................................................................

def CoffeeRecommender(coffee_name, df):
    # Kahve adına karşılık gelen kahve indeksini bulma
    coffee_id = df[df['drink'] == coffee_name].index
    coffee_id = coffee_id[0]

    # MinMaxScaler kullanarak özellikleri ölçeklendirme
    min_max_scaler = MinMaxScaler()
    features_scaled = min_max_scaler.fit_transform(df[['Caffeine (mg)', 'Calories', 'average_rating']])

    # NearestNeighbors modelini oluşturma
    model = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
    model.fit(features_scaled)

    # Modeli kullanarak komşu noktaları ve mesafeleri hesaplama
    distances, neighbors = model.kneighbors(features_scaled)

    # Kahvenin önerilmesi için komşu kahveleri bulma
    coffee_list_name = []
    for newid in neighbors[coffee_id]:
        coffee_list_name.append(df.loc[newid, 'drink'])

    return coffee_list_name

# Örnek
recommended_coffees = CoffeeRecommender('Zola Coconut Water Espresso', df)
print(recommended_coffees)
