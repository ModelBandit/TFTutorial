import tensorflow as tf #TF_ENABLE_ONEDNN_OPTS=0
import tensorflow.python.keras as pyTF
#from tensorflow.python.keras.layers import Dense


def Run():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0


    model = pyTF.models.Sequential([
        pyTF.layers.Flatten(input_shape=(28, 28)),
        pyTF.layers.Dropout(0, 2),
        pyTF.layers.Dense(128, activation='relu'),
        pyTF.layers.Dense(10, activation='softmax')
    ])
    print(model)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 예측값(10배열) 반환됨 당장은 초기값. 손실값 계산 H(q,p) 에서 p에 해당함
    predictions = model(x_train[:1]).numpy()
    print(predictions)

    tf.nn.softmax(predictions).numpy()# 안정적인 손실계산을 제공하지 않아서 권장하지 않음
    #그럼 왜 있음?

    #손실값 계산
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    print(f"before: {loss_fn}")
    loss_fn(y_train[:1], predictions).numpy()
    print(f"after: {loss_fn}")

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=1)

#Like C#
mnist = 0
x_train, y_train, x_test, y_test = 0,0,0,0
def DatasetLoad():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

#request cpu upgrade
# Intel core i7 9750H 2.6GHz 6core hyper threading

model = 0
def MachineRunningModelBuild():
    model = pyTF.keras.models.Sequential([
        pyTF.keras.layers.Flatten(input_shape=(28, 28)),
        pyTF.keras.layers.Dropout(0, 2),
        pyTF.keras.layers.Dense(128, activation='relu'),
        pyTF.keras.layers.Dense(10, activation='softmax')
    ])
    print(model)

    m = model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(m)



predictions = 0
def MatrixCalculate():
    predictions = model(x_train[:1]).numpy()# Dont using alone (using with {model})
    # logits와 log-odds 반환
    '''
    logits
    분류 모델에서 생성하는 원시 (정규화되지 않음) 예측의 벡터로, 일반적으로 정규화 함수에 전달됩니다.
    모델이 다중 클래스 분류 문제를 해결하는 경우 로짓은 일반적으로 소프트맥스 함수의 입력이 됩니다.
    그런 다음 소프트맥스 함수는 가능한 각 클래스에 대해 하나의 값을 갖는 (정규화된) 확률 벡터를 생성합니다.
    https://developers.google.com/machine-learning/glossary?hl=ko#logits

    log-odds
    특정 사건의 확률 로그입니다.
    https://developers.google.com/machine-learning/glossary?hl=ko#log-odds
    '''
    print(predictions)