import json
import plotly
import pandas as pd
import seaborn as sns


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
#from plotly.graph_objs import Bar
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:////home/workspace/data/DisasterResponse_Database.db')
df = pd.read_sql_table('DisasterResponse_Database', engine)
# load model
model = joblib.load('/home/workspace/models/my_classifier.pkl')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #extract all the disaster categories
    category_names = df.iloc[:,4:].columns
    #compute the number of each category disaster
    category_boolean = df.iloc[:,4:].sum().values
    
    #extract the list names of disaster categories
    my_list=list(df.iloc[:,4:].columns)
    
    #extract correlation matrix of the first 18 disaster categories
    cor0_17=df.iloc[:,4:][my_list[0:18]].corr()
    cor0_17=cor0_17.values.tolist()
                
    #extract correlation matrix of the firs 18 disaster categories
    cor18_34=df.iloc[:,4:][my_list[18:35]].corr()
    cor18_34=cor18_34.values.tolist()
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        
        # Graph1: Disaster Message genre type distributions
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Disaster Messages genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        #Graph2: Disaster Categories distributions
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean
                )
            ],

            'layout': {
                'title': 'Distribution of Disaster Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Class",
                    'tickangle': 35
                }
            }
        }
        
        
        ,
        {
         
          'data' : [
  Heatmap(
    z=cor0_17,
    x=my_list[0:18],
    y= my_list[0:18]
  )
],
            'layout': {
                'title': 'Heatmap Correlation of the first 18 disaster categories',
            }
        },
              {
         
          'data' : [
  Heatmap(
    z=cor18_34,
    x=my_list[18:35],
    y= my_list[18:35]
  )
],
                  
            'layout': {
                'title': 'Heatmap Correlation of the last 17 disaster categories',
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()