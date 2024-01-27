# %%
import pandas as pd

# %%
movies_df = pd.read_csv("data/IMDB-Movie-Data.csv", index_col='Title')

# %%
movies_df.head()

# %%
movies_df.tail(8)

# %%
movies_df.info()

# %%
movies_df.shape

# %%
# append is deprecated thus we are using pandas concat
temp_df = pd.concat([movies_df, movies_df])
temp_df.shape

# %%
# removing duplicates
temp_df = temp_df.drop_duplicates()
temp_df.shape

# %%
temp_df = pd.concat([movies_df, movies_df])
temp_df.drop_duplicates(inplace=True, keep=False)
temp_df.shape

# %%
movies_df.columns

# %%
movies_df.rename(columns={
 'Runtime (Minutes)': 'Runtime',
 'Revenue (Millions)': 'Revenue_millions'
 }, inplace=True)
movies_df.columns

# %%
# renaming using list comprehension
movies_df.columns = [col.lower() for col in movies_df]
movies_df.columns

# %%
movies_df.isnull().sum()

# %%
movies_df.dropna().shape

# %%
movies_df.dropna(axis=1).shape

# %%
subset = movies_df[['genre', 'rating']]
subset.head()

# %%
prom = movies_df.loc["Prometheus"]
prom

# %%
prom = movies_df.iloc[1]

# %%
movie_subset = movies_df.loc['Prometheus':'Sing']
movie_subset

# %%
condition = (movies_df['director'] == "Ridley Scott")
condition.head()

# %%
# movies_df[movies_df['director'] == "Ridley Scott"]
# since we defined it in condition we can just use condition
movies_df[condition]

# %%
def rating_function(x):
    if x >= 8.0:
        return "good"
    else:
        return "bad"
    
movies_df["rating_category"] = movies_df["rating"].apply(rating_function)
movies_df.head(2)

# %%
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20, 'figure.figsize': (10, 8)}) # set font and plot size to be larger

# %%
movies_df.plot(kind='scatter', x='rating', y='revenue_millions', title='Revenue (millions) vs Rating')

# %%
movies_df['rating'].plot(kind='hist', title='Rating')

# %%
movies_df.describe()

# %%



