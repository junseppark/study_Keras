from keras import layers, models

def ANN_models_func(Nin, Nh, Nout):
    x = layers.Input(shape=(Nin,)) # 입력 계층, 원소 Nin개
    h = layers.Activation('relu')(layers.Dense(Nh)(x)) # 은닉 계층, 노트 Nh개
    y = layers.Activation('softmax')(layers.Dense(Nout)(h))

    model = models.Model(x,y)
    model.compile(loss='categorical_crossvalidation', optimizer='adam', metrics=['accuracy'])
    return model
