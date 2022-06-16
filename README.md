# Denoise module

Модуль производит отчистку посторонних шумов в звуковом сигнале.

### Структура проекта:
common - содержит скрипты общие для всего проекта.
* data.py - преобразование данных и создание датасетов.
* model.py - содержит класс, определяющий модель для обучения.
* utils.py - содержит вспомогательные функции.

config - содержит конфигурационные настройки.

data - данные.

models - сохраненные обученные модели. 

predict - содержит класс, реализующий применение модели на тестовых данных. 

train - содержит класс реализующий обучение модели.


### Использование:
Перед использованием данного модуля нужно установить пакеты из ```requirements.txt```:
    
    pip install -r requirements.txt

Данные для обучения и валидации нужно разместить в папках ```data\train``` и  ```data\valid```, соответственно.

Для тестирования модели на новых данных необходимо расположить их в папке ```data\test```.

Чтобы запустить обучение модели необходимо запустить файл ```run_train.py```.

Чтобы протестировать модель нужно запустить файл ```run_predict.py```.
