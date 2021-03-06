from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#from keras.optimizers import Adam
#from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
#from utils import INPUT_SHAPE, batch_generator

from data_utils import load_image_data, split_train_test
from img_utils import process_image

if __name__ == '__main__':
    data, labels = load_image_data()
    X_train, y_train, X_test, y_test = split_train_test(data, labels, 0.6)

    pipe_line = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('model', RandomForestClassifier())
    ])

    pipe_line.fit_transform(X_train.astype(float), y_train)

    y_test_pred = pipe_line.predict(X_test.astype(float))
    print(confusion_matrix(y_test, y_test_pred))

    joblib.dump(pipe_line, 'car_detection.pkl')
    




#def build_model():
#    model = Sequential()
#    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
#    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
#    model.add(Dense(100, activation='elu'))
#    
#    model.add(Conv2D(64, 3, 3, activation='elu',subsample=(2, 2)))
#    model.add(Dense(50, activation='elu'))
#    
#    model.add(Conv2D(64, 3, 3, activation='elu',subsample=(1, 1)))
#    model.add(Dense(10, activation='elu'))
#    model.add(Dense(1))
#    model.add(Dropout(0.2))
#    model.add(Flatten())
#    model.summary()
#    adam = Adam(lr=1e-6)
#    model.compile(loss='mse',optimizer=adam)
#
#    return model


