import pandas as pd
import numpy as np
import tensorflow as tf
import math
import os
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Subtract, Lambda, Concatenate, multiply
from keras.losses import mean_squared_error
from tensorflow.keras.backend import print_tensor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def sparseEmbed(df, name, num, colIdx):
    embedName = [ name+"_"+str(i) for i in range(num)] 
    Emptydf = pd.DataFrame()
    Emptydf[embedName] = df[name].str.split('|',expand=True)
    values = np.unique(Emptydf[embedName].values)
    
    dic = {}
    a = 0
    for i in values:
        dic[i] = a
        a += 1
    dic.pop('nan', None)
    
    
    appendValue = np.zeros([Emptydf.values.shape[0], len(values)])
    for i in range(Emptydf.values.shape[0]):
        for j in range(num):
            key = Emptydf.values[i][j]
            if key in dic:
                appendValue[i][dic[key]] = 1
    
    for i in range(appendValue.shape[1], 0, -1):
        df.insert(colIdx, name+"_"+str(i-1), appendValue[:, i-1])
    
    del df[name]
    return df

def toDummy(df, name, colIdx):
    num = len(np.unique(df[name].values.astype(str)))-1
    embedName = [ name+"_"+str(i) for i in range(num)]  # don't need nan value
        
    dic = {}
    a = 0
    for i in range(num+1):
        dic[i] = a
        a += 1
    dic.pop('nan', None)
        
    appendValue = np.zeros([df[name].size, a])
    for i in range(df[name].size):
        key = df[name].values[i]
        if key in dic:
            appendValue[i][dic[key]] = 1
    
    for i in range(appendValue.shape[1], 0, -1):
        df.insert(colIdx, name+"_"+str(i-1), appendValue[:, i-1])
    
    del df[name]
    return df

def genderDummy(df, name, colIdx):
    pool = set()
    num = len(np.unique(df[name].values))-1
    for i in df[name].values:
        pool.add(str(i))
    num = len(list(pool))-1
    embedName = [ name+"_"+str(i) for i in range(num)]  # don't need nan value
        
    dic = {}
    a = 0
    for i in range(num+1):
        dic[i] = a
        a += 1
    dic.pop('nan', None)
        
    appendValue = np.zeros([df[name].size, a])
    for i in range(df[name].size):
        key = df[name].values[i]
        if key in dic:
            appendValue[i][dic[key]] = 1
    
    for i in range(appendValue.shape[1], 0, -1):
        df.insert(colIdx, name+"_"+str(i-1), appendValue[:, i-1])
    
    del df[name]
    return df


def main()：
    head = ["user_age", "user_gender", "user_7_hero", "user_30_hero", "user_7_keyword", "user_7_author", "item_rate", "item_keyword", "item_author", "item_avgTime", "item_numReader", "item_numTime", "label"]
    raw = pd.read_csv("./thing.txt", names=head, sep=",", index_col = False)

    colIdx = raw.columns.values.tolist().index("user_gender")
    raw = genderDummy(raw, "user_gender", colIdx)
    colIdx = raw.columns.values.tolist().index("item_keyword")
    raw = toDummy(raw, "item_keyword", colIdx)

    numDic = {"user_gender": 1, "user_7_hero": 5, "user_30_hero": 5, "user_7_keyword": 3, "user_7_author": 3, "item_keyword": 1, "item_author": 3}
    for i in ["user_7_hero", "user_30_hero", "user_7_keyword", "user_7_author", "item_author"]:
        colIdx = raw.columns.values.tolist().index(i)
        raw = sparseEmbed(raw, i, numDic[i], colIdx)
        print("finished with", i)

    # normalize numerical features into interval [0, 1]
    for i in ["user_age", "item_rate", "item_avgTime", "item_numReader", "item_numTime"]:
        r = raw[i].values.astype(float)
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(r.reshape(-1,1))
        raw_normalized = pd.DataFrame(x_scaled)
        raw[i] = raw_normalized

    raw = raw.sample(200000)
    
    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    batch = 1024

    data = raw.sample(50000)

    # Splitting dataframe into train, validation, and testing
    dataY = data['label'].values
    dataX = data.drop(columns = 'label').values


    X, Xtest, Y, Ytest = train_test_split(dataX, dataY, test_size = 0.2, random_state = 42)
    Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size = 0.25, random_state = 42)


    break_index = data.columns.values.tolist().index("item_rate") # first item index-1 is the break index
    length_total = data.values.shape[1]
    length_p = break_index # index of last user feature into length of the user feature
    length_g = length_total-length_p-1


    def pgSplit(data, idx):
        data_p = data[:, :idx]
        data_g = data[:, idx:]
        return data_p, data_g

    Xtrain_p, Xtrain_g = pgSplit(Xtrain, break_index)
    Xval_p, Xval_g = pgSplit(Xval, break_index)
    Xtest_p, Xtest_g = pgSplit(Xtest, break_index)

    a = 1
    global num_encode_1
    global num_encode_2
    global num_encode_3
    global num_neck
    global num_decode_1
    global num_decode_2
    global num_decode_3
    global num_output_to_p
    global num_output_to_g
    global threshold

    num_encode_1 = int(256 *a)
    num_encode_2 = int(128 *a)
    num_encode_3 = int(64 *a)
    num_neck = 8
    num_decode_1 = num_encode_3
    num_decode_2 = num_encode_2
    num_decode_3 = num_encode_1
    num_output_to_p = length_p
    num_output_to_g = length_g
    threshold = 0.5 * math.sqrt(num_neck)
    
    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    
    label = Input(shape=(1,))

    ## person autoencoder
    main_p_input = Input(shape=(length_p,))
    encode_p_1 = Dense(num_encode_1, activation='relu')(main_p_input)
    encode_p_2 = Dense(num_encode_2, activation='relu')(Dropout(0.05)(encode_p_1))
    encode_p_3 = Dense(num_encode_3, activation='relu')(Dropout(0.05)(encode_p_2))
    encode_p_neck = Dense(num_neck, activation= 'sigmoid')(encode_p_3) ###
    decode_p_1 = Dense(num_decode_1, activation='relu')(Dropout(0.05)(encode_p_neck))
    decode_p_2 = Dense(num_decode_2, activation='relu')(Dropout(0.05)(decode_p_1))
    decode_p_3 = Dense(num_decode_3, activation='relu')(Dropout(0.05)(decode_p_2))

    ## goods autoencoder
    main_g_input = Input(shape=(length_g,))
    encode_g_1 = Dense(num_encode_1, activation='relu')(main_g_input)
    encode_g_2 = Dense(num_encode_2, activation='relu')(Dropout(0.05)(encode_g_1))
    encode_g_3 = Dense(num_encode_3, activation='relu')(Dropout(0.05)(encode_g_2))
    encode_g_neck = Dense(num_neck, activation= 'sigmoid')(encode_g_3) ###
    decode_g_1 = Dense(num_decode_1, activation='relu')(Dropout(0.05)(encode_g_neck))
    decode_g_2 = Dense(num_decode_2, activation='relu')(Dropout(0.05)(decode_g_1))
    decode_g_3 = Dense(num_decode_3, activation='relu')(Dropout(0.05)(decode_g_2))



    ###### Define 4 output layers
    # Reconstruction Layer person
    output_p_out = Dense(num_output_to_p, activation= 'sigmoid', name = "p")(decode_p_3)

    # Reconstruction Layer goods
    output_g_out = Dense(num_output_to_g, activation= 'sigmoid', name = "g")(decode_g_3)

    # Covariance Layer
    def CovLayer(X):
        n_rows = tf.cast(tf.shape(X)[0], tf.float32)
        X = X - (tf.reduce_mean(X, axis=0))
        cov = tf.matmul(X, X, transpose_a=True) / n_rows
        return tf.reshape(tf.reduce_mean(tf.matrix_set_diag(cov, tf.zeros(num_neck, tf.float32))), [1])

    concat_layer = Concatenate(axis=0)([encode_p_neck, encode_g_neck])
    covLayer = Lambda(CovLayer, name="cov")(concat_layer) # Just a scalar layer

    # Signed Distance Layer
    def DisLayer(distance):
        return tf.reshape(tf.norm(distance, axis=1), (-1,1))

    distance = Subtract()([encode_p_neck, encode_g_neck])
    disLayer = Lambda(DisLayer, name="dist")(distance)


    ###### Define 3 loss
    #loss 1: reconstruction loss for person
       ## MSE

    #loss 2: reconstruction loss for goods
       ## MSE

    #loss 3: covariance loss for Covariance Layer
    def covarianceLoss(zeroCovariance, CovLayer):
        return CovLayer - 0

    #loss 4: distance loss for Distance Layer
    def distanceLoss(label, disLayer):
        sign = 2*label-1
        return tf.reduce_mean(tf.maximum(0.0, 0.6*threshold+tf.multiply(sign, disLayer-threshold)))

    tf.keras.backend.print_tensor(
        tf.less_equal(disLayer, threshold),
        message='Shit is'
    )

    ## Metric 
    def AUC(label, disLayer):
        output = K.cast(tf.less_equal(disLayer, threshold), tf.float32)
        auc = tf.metrics.auc(output, label)[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc

    def Accuracy(label, disLayer):
        output = K.cast(tf.less_equal(disLayer, threshold), tf.float32)
        accuracy = tf.metrics.accuracy(output, label)[1]
        K.get_session().run(tf.local_variables_initializer())
        return accuracy

    losses = {"p": 'mse',
              "g": 'mse',
              "cov": covarianceLoss,
              "dist": distanceLoss}

    weights = {"p": 0.25,
              "g": 0.25,
              "cov": 0.2,
              "dist": 1}

    metric = {"dist": [AUC, Accuracy]}
    
    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    
    zero_train = np.zeros((Xtrain_p.shape[0],1))
    zero_val = np.zeros((Xval_p.shape[0],1))

    model = Model(inputs= [main_p_input, main_g_input, label], outputs = [output_p_out, output_g_out, covLayer, disLayer])
    model.compile(optimizer='Adam', loss=losses, loss_weights=weights, metrics = metric)
    model.fit([Xtrain_p, Xtrain_g, Ytrain], [Xtrain_p, Xtrain_g, zero_train, Ytrain], validation_data=([Xval_p, Xval_g, Yval], [Xval_p, Xval_g, zero_val, Yval]), epochs=70, batch_size=batch)

if __name__ == "__main__":
    main()