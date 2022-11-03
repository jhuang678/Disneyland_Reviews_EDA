######################################################################
import os
import string
from tabulate import tabulate as tab
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator
######################################################################

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 199)
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')

def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))

def review_data(data:object = pd.DataFrame()):
    print("Shape of the dataset:",data.shape, "\n")
    #print(data.describe(include='all'))
    print("Preview dataset entries:")
    # print(tab(data.head(), headers='keys', tablefmt='psql'), "\n")
    print(data.head())

def review_data_col(data:object = pd.DataFrame(), verbose = True):
    df_col = pd.DataFrame({'col_name': data.columns,
                           'dtype': data.dtypes.values,
                           'count': data.count(),
                           'unique': data.nunique().values,
                           'num_null': data.isnull().sum()
                           })
    if verbose:
        print(tab(df_col, headers='keys', tablefmt='psql'), "\n")
    return df_col

def plot_time_data(data, title="Line Chart", figsize:tuple = (15,15),
                   xlabel="x-axis(time)", ylabel="y-axis(value)", file_path = "figure.png"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = data.plot(title = title, figsize = figsize, fontsize=12, linestyle='dashdot', markersize = 3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=30)
    for i in range(len(data)):
        ax.text(i, data.iloc[i], data.iloc[i], horizontalalignment='center')
    plt.grid(color='grey', linestyle='--', linewidth=0.4)
    ax.legend()
    fig = plt.gcf()
    fig.savefig(file_path)
    plt.close(fig)
    print("Please see the line chart as " + file_path)

def plot_histogram(data:object, title:str = 'Histogram',bins:int = 10, figsize:tuple = (15,15),
                   xlabel:str="x-axis", ylabel:str="y-axis(value)", file_path:str = "histogram.png"):
    mean = data.mean()
    std = data.std()
    ax = data.hist(bins = bins, figsize = figsize, edgecolor="k", align="mid", alpha=1, range= (1,6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=30)
    for i in range(1,len(data.value_counts())+1):
        ax.text(i, data.value_counts()[i], data.value_counts()[i], horizontalalignment='right')
    plt.axvline(mean, color='red', linestyle = 'dashed', linewidth = 2)
    #plt.axvline(mean + std, color='r', linestyle='dashed', linewidth=2)
    #plt.axvline(mean - std, color='r', linestyle='dashed', linewidth=2)
    fig = plt.gcf()
    fig.savefig(file_path)
    plt.close(fig)
    print("Please see the histogram as " + file_path)

def plot_bar_chart(data:object, title:str = 'Bar Chart', figsize:tuple = (15,15),
                   xlabel:str="x-axis", ylabel:str="y-axis", file_path:str = "bar_chart.png"):
    ax = data.plot.bar(rot=0, figsize = figsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title,fontsize=30)
    for i in range(len(data)):
        ax.text(i, data.iloc[i], data.iloc[i], horizontalalignment='center')
    ax.legend()
    fig = plt.gcf()
    fig.savefig(file_path)
    plt.close(fig)
    print("Please see the bar chart as " + file_path)

def plot_pie_chart(data:object, title:str = 'Bar Chart', figsize:tuple = (15,15), file_path:str = "pie_chart.png"):
    ax = data.plot.pie(figsize = figsize)
    ax.set_title(title,fontsize=30)
    for i in range(len(data)):
        ax.text(i, data.iloc[i], data.iloc[i], horizontalalignment='center')
    #ax.legend()
    fig = plt.gcf()
    fig.savefig(file_path)
    plt.close(fig)
    print("Please see the pie chart as " + file_path)

def plot_box_chart(x,y,data,file_path:str = "box_chart.png"):
    plt.figure(figsize=(15, 8))
    sns.boxplot(x=x, y=y, data=data)
    fig = plt.gcf()
    fig.savefig(file_path)
    plt.close(fig)
    print("Please see the box chart as " + file_path)

def plot_word_cloud(data,title:str = "Word Cloud", max_words:int = 50,
                    contour_color:str='red',file_path:str = "box_chart.png"):
    reviews = " ".join(review for review in data)
    fig, ax = plt.subplots(figsize=(20, 20))
    wordcloud_ALL = WordCloud(max_font_size=50, max_words=max_words,contour_color = contour_color,
                              background_color="white").generate(reviews)
    ax.imshow(wordcloud_ALL, interpolation='bilinear')
    ax.set_title(title, fontsize=40)
    ax.axis('off')
    fig = plt.gcf()
    fig.savefig(file_path)
    plt.close(fig)
    print("Please see the word cloud as " + file_path)

if __name__ == '__main__':
    print("------------------------------------------------------------------")
    print("[ Read Data Set ]")
    print(os.listdir('input/dlr'))
    data = pd.read_csv('input/dlr/DisneylandReviews.csv', encoding='latin-1')
    data['Year_Month'] = pd.to_datetime(data['Year_Month'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    data['Year'] = pd.DatetimeIndex(data['Year_Month']).year
    data['Month'] = pd.DatetimeIndex(data['Year_Month']).month

    print("Remove duplicated data......")
    data = data.drop_duplicates()
    data = data[~data.index.duplicated(keep='first')]
    review_data(data)

    print("------------------------------------------------------------------")
    print("[ Review Data Columns ]")
    review_data_col(data)

    print("------------------------------------------------------------------")
    print("[ Review Category Type Columns Counts]")

    print(data['Rating'].value_counts(), "\n")
    plot_histogram(data['Rating'], bins = 6, title = 'Histogram of Review Count by Rating', xlabel ="Rating",
                   ylabel ="Count of Review",  file_path = "histogram_of_review_count_by_rating.png")

    print(data['Branch'].value_counts(), "\n")
    plot_bar_chart(data['Branch'].value_counts(), title = 'Bar Chart of Review Count by Branch', xlabel ="Park Branch",
                   ylabel ="Count of Review",file_path = "bar_chart_of_review_count_by_branch.png")
    plot_pie_chart(data['Branch'].value_counts(), title = 'Pie Chart of Review Count by Branch',
                   file_path = "pie_chart_of_review_by_branch.png")

    table = pd.pivot_table(data, values='Rating', index=['Branch'], aggfunc=np.mean)
    sr_table = pd.Series(table.values.flatten().round(2), index = table.index, name='average_rating')
    plot_bar_chart(sr_table, title = 'Bar Chart of Average Rating by Branch', xlabel ="Park Branch",
                   ylabel ="Average Rating",
                   file_path = "bar_chart_of_average_rating_by_branch.png")
    plot_box_chart(x="Branch", y="Rating", data=data, file_path = "box_chart_of_rating_by_branch.png" )


    print(data['Reviewer_Location'].value_counts(), "\n")
    plot_bar_chart(data['Reviewer_Location'].value_counts().head(20), title = 'Bar Chart of Review Count by Location',
                   xlabel ="Country", ylabel ="Count of Review", figsize=(30,30),
                   file_path = "bar_chart_of_review_count_by_country.png")
    plot_pie_chart(data['Reviewer_Location'].value_counts(), title='Pie Chart of Review Count by Country',
                   file_path="pie_chart_of_review_by_country.png")

    print("------------------------------------------------------------------")
    print("[ Review Number Type Columns Counts]")
    print("Rating Category Counts")
    print(data['Rating'].describe(include='all'), "\n")

    print("------------------------------------------------------------------")
    print("[ Review Time Type Columns Counts]")
    print("Start Date:",data['Year_Month'].min())
    print("End Date:", data['Year_Month'].max())
    print("Number of Unique Days:", data['Year_Month'].nunique())
    print("Over Number of Years:", int(data['Year'].max() - data['Year'].min() + 1))
    plot_time_data(data['Year_Month'].value_counts(),
                   title = 'Time Chart of Number of Reviews', xlabel ="Time", ylabel ="Counts",
                   file_path = "time_chart_of_review_counts.png")

    print("\nCounts of Reviews by Year:")
    print(data['Year'].dropna().astype(int).value_counts(sort=False).sort_index())
    plot_bar_chart(data['Year'].dropna().astype(int).value_counts(sort=False).sort_index(),
                   title = 'Bar Chart of Counts of Reviews by Year', xlabel ="Year", ylabel ="Number of Reviews",
                   file_path = "bar_chart_of_total_reviews_by_year.png")

    print("\nCounts of Reviews by Month:")
    print(data['Month'].dropna().astype(int).value_counts(sort=False).sort_index())
    plot_bar_chart(data['Month'].dropna().astype(int).value_counts(sort=False).sort_index(),
                   title='Bar Chart of Counts of Reviews by Month', xlabel="Month", ylabel="Number of Reviews",
                   file_path="bar_chart_of_total_reviews_by_month.png")

    print("------------------------------------------------------------------")
    print("[ Generate Statistical Count Features ]")
    data_text = data.copy()
    data_text['word_count'] = data_text['Review_Text'].apply(lambda x: len(x.split()))
    data_text['char_count'] = data_text['Review_Text'].apply(lambda x: len(x.replace(" ", "")))
    data_text['word_density'] = data_text['word_count'] / (data_text['char_count'] + 1)
    data_text['punc_count'] = data_text['Review_Text'].apply(lambda x: len([a for a in x if a in string.punctuation]))
    print(tab(data_text[['word_count', 'char_count', 'word_density', 'punc_count']].head(10), headers='keys',
              tablefmt='psql'), "\n")

    print("------------------------------------------------------------------")
    print("[ Wordcloud Visualization ]")

    print("\nWord Cloud of Positive Reviews:")
    print(data_text[data_text['Rating'] == 5]['Review_Text'].head())
    plot_word_cloud(data = data_text[data_text['Rating'] == 5]['Review_Text'],
                    title='Word Cloud of All Rated 5 Reviews', file_path="word_cloud_of_positive_reviews.png")

    print("\nWord Cloud of Negative Reviews:")
    print(data_text[data_text['Rating'] == 1]['Review_Text'].head())
    plot_word_cloud(data=data_text[data_text['Rating'] == 1]['Review_Text'],
                    title='Word Cloud of All Rated 1 Reviews', file_path="word_cloud_of_negative_reviews.png")
    print('\nFinished!')
