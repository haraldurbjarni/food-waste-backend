# importing the required libraries
from flask import jsonify, Flask, render_template, request, redirect, url_for
from flask_cors import CORS, cross_origin
from numpy import genfromtxt
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional
from keras.layers import Dense
import numpy as np
import os
import io
import tensorflow as tf

dir_path = os.path.dirname(os.path.realpath(__file__))

# start flask
app = Flask(__name__)
hashmap = {}
CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


if not os.path.exists('./files'):
    os.makedirs('./files')

# render default webpage
@app.route('/api/upload_data',  methods=['POST'])
@cross_origin(supports_credentials=True)
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        random_string = os.urandom(10).hex() 
        hashmap[random_string] = f
        f.save(f'./files/{random_string}.csv')
        print('Data key set',random_string,':',f)

        perday_array = np.genfromtxt(f'./files/{random_string}.csv', delimiter=",")[1:,1:]
        print(perday_array)
        df = pd.read_csv(f'./files/{random_string}.csv') 
        dates = list(df['Dags reiknings'])
        sales_list = []
        perday_totals = []
        for i in range(perday_array.shape[0]):
            sales_list.append(np.sum(perday_array[i]))
            perday_totals.append({'name': dates[i],'data':sales_list[i]})
        
        product_list = []
        products = list(df.head())[1:]
        product_totals = []
        for i in range(perday_array.shape[1]):
            product_list.append((np.sum(perday_array[:,i]),i))
        product_list.sort(reverse=True)
        for i in range(10):
            product_totals.append({'name':product_list[i][0], 'data': products[product_list[i][1]]})
        print(product_totals)

        return jsonify({'key': random_string,
                        'Perday totals': perday_totals,
                        'Product totals': product_totals}), 200, {'ContentType':'application/json'} 

@app.route('/api/upload_prices',  methods=['POST'])
def upload_prices():
    if request.method == 'POST':
        p = request.files['file']
        random_string = os.urandom(10).hex()
        hashmap[random_string] = p
        p.save(f'./files/{random_string}.csv')
        print('Price key set',random_string,':',p)
        return jsonify({'key': random_string}), 200, {'ContentType':'application/json'} 
        #IMPORTANT
        #price_list = pd.read_csv(p, delimiter=',')
        #price_dict = dict(sorted(price_list.values.tolist()))
        #session.get('price_list') = price_list

@app.route('/api/test',  methods=['GET'])
@cross_origin(supports_credentials=True)
def test():
    print('Hallo', request.args.get('key'))
    print(hashmap[request.args.get('key')])
    return jsonify(session.get('pandas_data'))

@app.route('/api/train_model',  methods=['GET'])
@cross_origin(supports_credentials=True)
def train_model():
    #getting the data from the requests
    data_key = request.args.get('dataKey')
    price_key = request.args.get('priceKey')
    profit_margin = request.args.get('profitMargin')
    model_type = request.args.get('modelType')

    with open(f'./files/{data_key}.csv','rb') as d:
        pd_data =  pd.read_csv(d, delimiter=',',encoding='utf-8')
        d.close()
    with open(f'./files/{data_key}.csv','r') as f:
        ml_data = genfromtxt(f, delimiter=',')
        ml_data = ml_data[1:,1:]
        f.close()
    with open(f'./files/{price_key}.csv','r') as p:
        price_list = pd.read_csv(p, delimiter=',',encoding='utf-8')
        price_dict = dict(sorted(price_list.values.tolist()))
        p.close()

    cols = list(pd_data.columns)[1:]
    model = create_model(model_type)
    n_steps = 14
    n_features = 1
    output_window = 7
    leave_out_number = 7
    prediction_array = np.zeros(shape=(len(cols)))
    actual_value_array = np.zeros(shape=(len(cols)))
    for i,item in enumerate(cols):
        data = ml_data[:,i]
        X, y = split_sequence_sum(data,output_window, n_steps)
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        X_train = X[:X.shape[0]-leave_out_number-1]
        y_train = y[:y.shape[0]-leave_out_number-1]
        X_val = X[X.shape[0]-1]
        X_val = np.expand_dims(X_val, axis=0)

        y_val = y[y.shape[0]-1]
        model.fit(X_train, y_train, epochs=200, verbose=0)
        y_pred = model.predict(X_val)
        y_pred = np.floor(y_pred[0])

        prediction_array[i] = y_pred[0]
        actual_value_array[i] = y_val

    prediction_array = np.array([0 if i<0 else i for i in list(prediction_array)])
    model_dict = {}
    total_sales_profit = 0
    total_capital_wasted = 0
    total_capital_misseed_out_on = 0
    for i in range(prediction_array.shape[0]):
        profit_margin = float(profit_margin)
        price = (price_dict[cols[i]])*profit_margin

        capital_wasted = 0
        capital_missed_out_on = 0
        if actual_value_array[i] > prediction_array[i]:
            sales_profit = prediction_array[i]*price
            capital_missed_out_on = (actual_value_array[i]-prediction_array[i])*price
        elif actual_value_array[i] < prediction_array[i]:
            sales_profit = actual_value_array[i]*price
            capital_wasted = (prediction_array[i] - actual_value_array[i])*(price_dict[cols[i]]*(1-profit_margin))
        else:
            sales_profit = actual_value_array[i]*price
        total_sales_profit+=sales_profit
        total_capital_misseed_out_on+=capital_missed_out_on
        total_capital_wasted+=capital_wasted
        model_dict[cols[i]] = {
            'Predicted value': prediction_array[i],
            'Actual value': actual_value_array[i],
            'Price': price_dict[cols[i]],
            'Sales profit': sales_profit, 
            'Capital missed out on': capital_missed_out_on, 
            'Capital wasted': capital_wasted
        }
    model.save(f'./models/{data_key}')
    return jsonify(model_dict, {'Total sales profit':total_sales_profit,
                                'Total capital missed out on': total_capital_misseed_out_on,
                                 'Total capital wasted': total_capital_wasted}
                                 )


@app.route('/api/make_prediction',  methods=['GET'])
@cross_origin(supports_credentials=True)
def make_prediction():
    data_key = request.args.get('dataKey')
    with open(f'./files/{data_key}.csv','r') as f:
        ml_data = genfromtxt(f, delimiter=',')
        ml_data = ml_data[1:,1:]
        f.close()
    with open(f'./files/{data_key}.csv','rb') as d:
        pd_data =  pd.read_csv(d, delimiter=',',encoding='utf-8')
        d.close()
    cols = list(pd_data.columns)[1:]
    loaded_model = tf.keras.models.load_model(f'./models/{data_key}')
    prediction_array = np.zeros(shape=(len(cols)))
    n_steps = 14
    n_features = 1
    output_window = 7
    leave_out_number = 7
    for i,item in enumerate(cols):
        data = ml_data[:,i]
        X, y = split_sequence_sum(data,output_window, n_steps)
        X_val = data[-14:]
        X_val = np.expand_dims(X_val, axis=0)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], n_features)

        y_pred = loaded_model.predict(X_val)
        y_pred = np.floor(y_pred[0])

        prediction_array[i] = y_pred[0]

    prediction_array = np.array([0 if i<0 else i for i in list(prediction_array)])
    model_dict = {}
    for i in range(prediction_array.shape[0]):
        model_dict[cols[i]] = {
            'Predicted value': prediction_array[i],
        }
    return jsonify(model_dict)





def create_model(type):
    #Model 1: LSTM
    model = Sequential()
    n_steps = 14
    n_features = 1
    output_window = 7
    if type == 'LSTM':
        model.add(LSTM(20, activation='relu', input_shape=(n_steps, n_features)))
    elif type == 'RNN':
        model.add(SimpleRNN(20, activation='relu', input_shape=(n_steps, n_features)))
    elif type == 'GRU':
        model.add(GRU(20, activation='relu', input_shape=(n_steps, n_features)))
    elif type == 'RNN':
        model.add(Bidirectional(LSTM(20, activation='relu', input_shape=(n_steps, n_features))))
    #model_2.add(LSTM(10, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    #model_2.add(LSTM(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def split_sequence_sum(sequence, output_window, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix  + output_window > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], np.sum(sequence[end_ix:end_ix+output_window])
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)