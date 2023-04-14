#!/usr/bin/env python
# coding: utf-8
# <div style="border:solid Chocolate 2px; padding: 40px">
#
# <b> Юрий, привет!👋</b>
#
# Меня зовут Кирилл Васильев, я буду ревьюером твоего проекта. Я предлагаю общаться на «ты», но если привычнее на «вы», просто скажи об этом!
#
# Я буду оставлять комментарии в твоем проекте. Пожалуйста, не удаляй их, они могут быть полезны в случае повторной проверки.
#
# Ты можешь реагировать на комментарии как просто написав их в Markdown-ячейках, так и выделив их цветом. Например, <font color='blue'>синим</font>. Второй способ, на мой взгляд, удобнее.
#
# Свои комментарии я буду обозначать <font color='green'>зеленым</font>, <font color='gold'>желтым</font> и <font color='red'>красным</font> цветами, например:
#
# <br/>
#
# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#
# <b>Все отлично!👍:</b> В случае, если решение на отдельном шаге является полностью правильным. Здесь же я могу давать советы и предложения.
# </div>
#
# <br/>
#
# <div class="alert alert-warning">
#     <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#
# <b>Некоторые замечания и рекомендации💡:</b> В случае, когда решение на отдельном шаге станет еще лучше, если внести небольшие коррективы.
# </div>
#
# <br/>
# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
#
#
# <b>На доработку🤔:</b>
#  В случае, когда решение на отдельном шаге требует существенной переработки и внесения правок. Напоминаю, что проект не может быть принят с первого раза, если ревью содержит комментарии, рекомендующие доработать шаги.
# </div>
#
# Увидев у тебя неточность, в первый раз я лишь укажу на ее наличие и дам тебе возможность самому найти и исправить ее. На реальной работе твой руководитель будет поступать также. Но если ты пока не справишься с такой задачей - при следующей проверке я дам более точную подсказку!

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка данных</a></span></li><li><span><a href="#Исследование-задачи" data-toc-modified-id="Исследование-задачи-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Исследование задачи</a></span></li><li><span><a href="#Борьба-с-дисбалансом" data-toc-modified-id="Борьба-с-дисбалансом-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Борьба с дисбалансом</a></span></li><li><span><a href="#Тестирование-модели" data-toc-modified-id="Тестирование-модели-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Тестирование модели</a></span></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Чек-лист готовности проекта</a></span></li></ul></div>

# # Отток клиентов

# Из «Бета-Банка» стали уходить клиенты. Каждый месяц. Немного, но заметно. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых.
#
# Нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. Вам предоставлены исторические данные о поведении клиентов и расторжении договоров с банком.
#
# Постройте модель с предельно большим значением *F1*-меры. Чтобы сдать проект успешно, нужно довести метрику до 0.59. Проверьте *F1*-меру на тестовой выборке самостоятельно.
#
# Дополнительно измеряйте *AUC-ROC*, сравнивайте её значение с *F1*-мерой.
#
# Источник данных: [https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling](https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling)

# ## Подготовка данных

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

data = pd.read_csv('/datasets/Churn.csv')
data.info()
data.head(20)


# В столбце "Tenure — сколько лет человек является клиентом банка" имеются пропуски, почти 10%.
# Мы можем:
# удалить пропуски
# вставить в промуски медианное значени
# вставить в промуски минимальное значени
#
# Так как пропуски имею большое количество от общей массы, я считаю что лучше их удалить, так как они могут создать сильный перевес в случаи их заполение например минимальным значением
#

# In[2]:


data = data.dropna(subset=['Tenure'])


# Рассмотрим далее описательную статистику, посте заполения пропусков
#

# In[3]:


data.describe()


# In[4]:


data.describe(include='all')


#  Проверим на дублирование данных

# In[5]:


data.duplicated().sum()


# Дублирование данных отсутствует

# In[6]:


print(data['Geography'].unique())
print(data['Gender'].unique())


# Проверим значения на опечатки и разность в написании. они отсутствуют
#

# Вывод
# Мы расмоотрели данные и описательную статистику, нашли пропуски в столбце Tenure и удалили их что бы они не влияли на результаты иследования. В результатте из 10 000 у нас осталось 9091 строк. Так же провери данные на дубли - они отсутствуют и рассмотрели уникальные значения на наличие опечаток или задвоений, они также отсутствуют.

# ## Исследование задачи

# Удалим лишнии данные, которые нам не пригодятся

# In[7]:


drop = ['RowNumber', 'CustomerId', 'Surname']
data = data.drop(drop, axis=1)
data.head()


# Используем метод OHE для избежания дамми ловушки

# In[8]:


data = pd.get_dummies(data, drop_first=True)
data.head()


# Разделим данные на 2 выборки на признаки и целевой признак

# In[9]:


features = data.drop('Exited', axis=1)
target = data['Exited']


# Создатим 3 выборки
# Обучвющая - 60% features_train и target_train
# Валидационная - 20% features_valid и target_valid
# Тестовая - 20% features_test и target_test
#
#
#
#
#

# In[10]:


features_train, features_validtest, target_train, target_validtest = train_test_split(
    features, target, train_size=0.6, random_state=12345)
features_valid, features_test, target_valid, target_test = train_test_split(
    features_validtest, target_validtest, train_size=0.5, random_state=12345)


# Масштабируем числ признаки

# In[11]:


numeric = ['CreditScore', 'Age', 'Tenure',
           'Balance', 'NumOfProducts', 'EstimatedSalary']
scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])
print('Масштабируем числ признаки обучающей выборки\n', features_train.head())
features_valid[numeric] = scaler.transform(features_valid[numeric])
print('Масштабируем численные признаки валидационной выборки \n',
      features_valid.head())
features_test[numeric] = scaler.transform(features_test[numeric])
print('Масштабируем численные признаки тестовой выборки \n', features_test.head())


# Созданим функию для метрик

# In[12]:


def f1(target_valid, prediction):
    print("Полнота", recall_score(target_valid, prediction))
    print("Точность", precision_score(target_valid, prediction))
    print("F1-мера", f1_score(target_valid, prediction))


# Рассмотри модель дерево решений

# In[13]:


f1_res = 0
for depth in range(1, 20, 1):
    model_DTC = DecisionTreeClassifier(max_depth=depth, random_state=12345)
    model_DTC.fit(features_train, target_train).score(
        features_valid, target_valid)
    model_DTC_prediction = model_DTC.predict(features_valid)

    res = f1_score(target_valid, model_DTC_prediction)
    if f1_res < res:
        p = model_DTC_prediction
        f1_res = res

        depth_model = depth
    print('depth', depth, 'F1:', f1_score(target_valid, model_DTC_prediction))
print('Лучший результат depth', depth_model, 'F1:', f1_score(target_valid, p))


# Лучшие показатели при F1- 0.5764331210191083 при 7 деревьев

# Рассмотрим модель логической регресси

# In[14]:


model_LG = LogisticRegression(solver='liblinear')
model_LG.fit(features_train, target_train).score(features_valid, target_valid)
model_LG_prediction = model_LG.predict(features_valid)
print(f1(target_valid, model_LG_prediction))


# Расмотрим модель случайны лес

# In[15]:


f1_res = 0
for est in range(10, 51, 10):
    for depth in range(1, 12):
        model_RFC = RandomForestClassifier(
            n_estimators=est, max_depth=depth, random_state=12345)
        model_RFC.fit(features_train, target_train)
        model_RFC_prediction = model_RFC.predict(features_valid)
        res = f1_score(target_valid, model_RFC_prediction)
        if f1_res < res:
            p = model_RFC_prediction
            f1_res = res
            est_model = est
            depth_model = depth
        print('estim', est, 'depth', depth, 'F1:',
              f1_score(target_valid, model_RFC_prediction))
print('--------------------------------')
print('Лучший результат estim', est_model, 'depth', depth_model)
print(f1(target_valid, p))


# Вывод
# Мы рассматрели 3 модели
#  дерево решений Лучший результат depth 7 F1: 0.5764331210191083
#  логическая регресия F1-мера 0.30400000000000005
#  случайный лес Лучший результат estim 40 depth 11 F1-мера 0.5796610169491526
#
# По итогу лучший результат показал случайный лес, дерево решений не намного отстало

# ## Борьба с дисбалансом

# Увеличим выборку редких объектов  с помощью техникик апсемплинг

# In[17]:


def upsampling(features, target, repeat):
    # Разделяем выборку по значениям целевой функции
    target_one = target[target == 1]
    target_null = target[target == 0]
    features_one = features[target == 1]
    features_null = features[target == 0]

    # Увеличиваем и соединяем обратно
    upsampling_features = pd.concat([features_null]+[features_one]*repeat)
    upsampling_target = pd.concat([target_null]+[target_one]*repeat)

    # Перемешиваем
    upsampling_features, upsampling_target = shuffle(
        upsampling_features, upsampling_target, random_state=1234)

    return upsampling_features, upsampling_target


features_upsampled, target_upsampled = upsampling(
    features_train, target_train, 4)
print(target_upsampled.shape)


# поправил

# In[ ]:


# Уменьши выборку частых объектов  с помощью техникик даунсемплинг

# In[18]:


def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat([features_zeros.sample(
        frac=fraction, random_state=12345)] + [features_ones])
    target_downsampled = pd.concat([target_zeros.sample(
        frac=fraction, random_state=12345)] + [target_ones])

    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345)

    return features_downsampled, target_downsampled


features_downsampled, target_downsampled = downsample(
    features_train, target_train, 0.25)
print(target_downsampled.shape)


# Вывод:
# Так как количество данных при использовании Downsampling крайне мало, мы его не будем использовать
#

# вчера этот раздел был полностью сделан, открываю с утра и всё пропало.... уж второй раз так...

# рассмотрим модель дерево решений после балансирования данных

# In[19]:


f1_res = 0
for depth in range(1, 20, 1):
    model_DTC_upsampled = DecisionTreeClassifier(
        max_depth=depth, random_state=12345)
    model_DTC_upsampled.fit(features_upsampled, target_upsampled)
    model_DTC_upsampled_prediction = model_DTC_upsampled.predict(
        features_valid)

    res = f1_score(target_valid, model_DTC_upsampled_prediction)
    if f1_res < res:
        p = model_DTC_upsampled_prediction
        f1_res = res

        depth_model = depth
print('Лучший результат upsampled - depth',
      depth_model, 'F1:', f1_score(target_valid, p))


f1_res = 0
for depth in range(1, 20, 1):
    model_DTC_downsampled = DecisionTreeClassifier(
        max_depth=depth, random_state=12345)
    model_DTC_downsampled.fit(features_downsampled, target_downsampled)
    model_DTC_downsampled_prediction = model_DTC_downsampled.predict(
        features_valid)

    res = f1_score(target_valid, model_DTC_downsampled_prediction)
    if f1_res < res:
        p = model_DTC_downsampled_prediction
        f1_res = res

        depth_model = depth
print('Лучший результат downsampled - depth',
      depth_model, 'F1:', f1_score(target_valid, p))


# результаты почти не изменились прошлый результат был depth 7 F1: 0.5764331210191083. Модель стабильная

# Рассмотрим модель случайный лес

# In[20]:


f1_res = 0
for est in range(10, 51, 10):
    for depth in range(1, 12):
        model_RFC_upsampled = RandomForestClassifier(
            n_estimators=est, max_depth=depth, random_state=12345)
        model_RFC_upsampled.fit(features_downsampled, target_downsampled)
        model_RFC_upsampled_prediction = model_RFC_upsampled.predict(
            features_valid)
        res = f1_score(target_valid, model_RFC_upsampled_prediction)
        if f1_res < res:
            p = model_RFC_upsampled_prediction
            f1_res = res
            est_model = est
            depth_model = depth
print('--------------------------------')
print('Лучший результат upsampled - estim', est_model, 'depth', depth_model)
print(f1(target_valid, p))

f1_res = 0
for est in range(10, 51, 10):
    for depth in range(1, 12):
        model_RFC_downsampled = RandomForestClassifier(
            n_estimators=est, max_depth=depth, random_state=12345)
        model_RFC_downsampled.fit(features_downsampled, target_downsampled)
        model_RFC_downsampled_prediction = model_RFC.predict(features_valid)
        res = f1_score(target_valid, model_RFC_downsampled_prediction)
        if f1_res < res:
            p = model_RFC_downsampled_prediction
            f1_res = res
            est_model = est
            depth_model = depth
print('--------------------------------')
print('Лучший результат downsampled -  estim', est_model, 'depth', depth_model)
print(f1(target_valid, p))


# Результаты при upsampled улучшелись. Прошлый результат estim 40 depth 11 F1-мера 0.5796610169491526

# рассмотрим логическую регрессию

# In[21]:


model_LG_upsampled = LogisticRegression(
    solver='liblinear', class_weight='balanced')
model_LG_upsampled.fit(features_upsampled, target_upsampled)
LG_prediction_upsampled = model_LG_upsampled.predict(features_valid)
print('Лучший результат upsampled - ',
      f1(target_valid, LG_prediction_upsampled))
print()
model_LG_downsampled = LogisticRegression(
    solver='liblinear', class_weight='balanced')
model_LG_downsampled.fit(features_downsampled, target_downsampled)
LG_prediction_downsampled = model_LG_downsampled.predict(features_valid)
print('Лучший результат downsampled - ',
      f1(target_valid, LG_prediction_downsampled))


# Результаты сильно улучшелись прошлый результат F1-мера 0.30400000000000005

# Вывод
# - случайный лес
#     Лучший результат upsampled F1-мера 0.6072874493927125
#     Лучший результат downsampled F1-мера 0.5660377358490567
#     Результаты при upsampled улучшелись, её в дальнейшем и возьмем для тестирования(estim 40 depth 8). Прошлый результат estim 40 depth 11 F1-мера 0.5796610169491526
#
# -дерево решений
#     Лучший результат upsampled - depth 5 F1: 0.5735449735449736
#     Лучший результат downsampled - depth 6 F1: 0.5636704119850188
#     результаты почти не изменились прошлый результат был depth 7 F1: 0.5764331210191083. Модель стабильная
#
# - логистическая регресия
#     Лучший результат upsampled F1-мера 0.509731232622799
#     Лучший результат downsampled F1-мера 0.5059360730593607
#
# Результаты сильно улучшелись прошлый результат F1-мера 0.30400000000000005

# ## Тестирование модели

# Протестируем модель дерево решений

# In[32]:


model = RandomForestClassifier(bootstrap=True, class_weight='balanced', max_depth=8,
                               n_estimators=40, random_state=12345)

model.fit(features_upsampled, target_upsampled)
predicted_test = model.predict(features_test)
probabilities_test = model.predict_proba(features_test)
probabilities_one_test = probabilities_test[:, 1]
fpr, tpr, thresholds = roc_curve(target_test, probabilities_one_test)


auc_roc = roc_auc_score(target_test, probabilities_one_test)
print('AUC-ROC', auc_roc)
plt.plot(fpr, tpr, label='RandomForestClassifier - balanced')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend(loc='lower right', fontsize='x-large')

plt.title("ROC-кривая")
plt.show()


print(f1(target_test, predicted_test))


# Сделаем константную модель

# Удалось достич.

# Вывод
# Нам  удалось достич цели в 0,59 по ф1. МЛучшая модель оказалась Случайный лес.
#
#
#
# Обученная модель случайного леса со взвешенными классами имеет достаточную адекватность, подтвержденная ее значением AUC-ROC = 0.861.
#
# Точность попадания по классам 0.861, precision 0.50, recall 0.71 и f1 0.59. Модель старается собрать больше данных, чем показать верную точность. Процент попадания в классы достаточно высокий.
#
#
#
#
