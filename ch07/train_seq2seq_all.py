import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt

from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq

# 하이퍼파라미터 설정
wordvec_size = 16
hideen_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0
acc_list = []
acc_list_reverse = []
acc_list_peeky = []

# baseline-----------------------------------------------------------
def do_seq2seq():
    print("--------------seq2seq----------------")

    # 데이터셋 읽기
    (x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
    char_to_id, id_to_char = sequence.get_vocab()

    # 입력 반전 여부 설정
    is_reverse = False

    if is_reverse:
        # 다음과 같이 표기함으로써 입력 데이터의 순서를 반전시킬 수 있음
        x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

    # 하이퍼파라미터 설정
    vocab_size = len(char_to_id)
    model = Seq2seq(vocab_size, wordvec_size, hideen_size)
    optimizer = Adam()  # 매개변수 최적화기법으로서 Adam 사용
    trainer = Trainer(model, optimizer)

    for epoch in range(max_epoch):
        # Trainer 클래스의 fit 메소드를 호출하여 학습 수행
        trainer.fit(x_train, t_train, max_epoch=1, 
                    batch_size=batch_size, max_grad=max_grad)
        correct_num = 0

        for i in range(len(x_test)):
            question, correct = x_test[[i]], t_test[[i]]
            verbose = i < 10
            
            # util.py의 eval_seq2seq 메소드 호출하여 덧셈 문제 평가 수행
            correct_num += eval_seq2seq(model, question, correct,
                                        id_to_char, verbose, is_reverse)
        acc = float(correct_num) / len(x_test)
        acc_list.append(acc)
        print('검증 정확도 %.3f%%' % (acc * 100))

# reverse-----------------------------------------------------------
# baseline에서 입력 데이터를 반전한 버전
def do_seq2seq_reverse():
    print("----------Reverse--------------")

    (x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
    char_to_id, id_to_char = sequence.get_vocab()

    is_reverse = True

    if is_reverse:
        x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

    vocab_size = len(char_to_id)
    model = Seq2seq(vocab_size, wordvec_size, hideen_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    for epoch in range(max_epoch):
        trainer.fit(x_train, t_train, max_epoch=1,
                    batch_size=batch_size, max_grad=max_grad)
        correct_num = 0

        for i in range(len(x_test)):
            question, correct = x_test[[i]], t_test[[i]]
            verbose = i < 10
            correct_num += eval_seq2seq(model, question, correct,
                                        id_to_char, verbose, is_reverse)
        acc = float(correct_num) / len(x_test)
        acc_list_reverse.append(acc)
        print('검증 정확도 %.3f%%' % (acc * 100))

# peeky-----------------------------------------------------------
# baseline에서 입력 데이터를 반전하고 엿보기 기법을 반영한 버전
def do_seq2seq_peeky():
    print("------------Peeky--------")

    (x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
    char_to_id, id_to_char = sequence.get_vocab()

    is_reverse = True

    if is_reverse:
        x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

    vocab_size = len(char_to_id)

    # 신경망 모델을 PeekySeq2seq로 설정
    model = PeekySeq2seq(vocab_size, wordvec_size, hideen_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    for epoch in range(max_epoch):
        trainer.fit(x_train, t_train, max_epoch=1,
                    batch_size=batch_size, max_grad=max_grad)
        correct_num = 0

        for i in range(len(x_test)):
            question, correct = x_test[[i]], t_test[[i]]
            verbose = i < 10
            correct_num += eval_seq2seq(model, question, correct,
                                        id_to_char, verbose, is_reverse)
        acc = float(correct_num) / len(x_test)
        acc_list_peeky.append(acc)
        print('검증 정확도 %.3f%%' % (acc * 100))

# baseline - seq2seq--------
do_seq2seq()

# seq2seq Reverse-----------
do_seq2seq_reverse()

# seq2seq Peeky-------------
do_seq2seq_peeky()

# 그래프 그리기, 하나의 그래프에 3개의 변수 추가
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o', label='baseline')

x_reverse = np.arange(len(acc_list_reverse))
plt.plot(x_reverse, acc_list_reverse, marker='o', label='reverse')

x_peeky = np.arange(len(acc_list_peeky))
plt.plot(x_peeky, acc_list_peeky, marker='o', label='reverse+peeky')

plt.legend(loc='upper left')   # 범례 위치를 좌상단으로 지정
plt.xlabel('epoch')            # x축 이름을 ‘epoch’로 지정 
plt.ylabel('accuracy')         # y축 이름을 ‘accuracy’로 지정 
plt.ylim(0, 1.0)               # y축 값의 범위를 0~1.0으로 지정
plt.show()