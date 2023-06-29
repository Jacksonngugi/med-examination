import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
def add_column (df):
    value =  df['weight']/(df['height']/100)**2

    if value <= 25:
        return 0
    
    else:
        return 1
    


df['overweight'] = df.apply( add_column, axis = 1)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'].mask(df['cholesterol'] == 1, 0, inplace=True)
df['cholesterol'].mask(df['cholesterol'] > 1, 1, inplace=True)
df['gluc'].mask(df['gluc'] == 1, 0, inplace=True)
df['gluc'].mask(df['gluc'] > 1, 1, inplace=True)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars =['cardio'], value_vars =['cholesterol','gluc','smoke','alco','active','overweight'])


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly
    vari = [*set((df_cat['variable']).tolist())]
    data = sorted(vari)
    value =  [*set((df_cat['value']).tolist())]
    card =  [*set((df_cat['cardio']).tolist())] 
    dt = pd.DataFrame(columns=['cardio','variable','count','value'])
    for i in data:
        for j in card:
          for v in value:
              d = df_cat[(df_cat['cardio'] ==  j) & (df_cat['variable'] == i) & (df_cat['value'] == v)]

              data = {'cardio': j,'variable':i,'value':v,'count':len(d)}

              dt.loc[len(dt.index)] = data

    # Draw the catplot with 'sns.catplot()'

    fig = sns.catplot(x='variable',y='count',data=dt,col='cardio',kind='bar',hue='value')

    # Get the figure for the output
    plt.show()


    # Do not modify the next two lines
    
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.drop(df[(df['ap_lo'] > df['ap_hi']) | (df['height'] < df['height'].quantile(0.025)) | (df['height'] > df['height'].quantile(0.975)) | (df['weight'] < df['weight'].quantile(0.025)) | (df['weight'] > df['weight'].quantile(0.975))].index)

    # Calculate the correlation matrix
    corr = df_heat.corr()
    print(corr)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(df_heat.corr()))



    # Set up the matplotlib figure
    fig, ax = plt.subplots()

    # Draw the heatmap with 'sns.heatmap()'

    sns.heatmap(corr, cmap="rocket", annot=True, mask=mask, fmt='.1f')

    plt.show()

    # Do not modify the next two lines
    
    return fig


draw_cat_plot()
draw_heat_map()
