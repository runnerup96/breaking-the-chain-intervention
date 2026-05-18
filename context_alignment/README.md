# Дообучение с помощью DPO

Чтобы дообучить модель, нужно выполнить следующий пайплайн:

## 1. Разбиение датасета на train, val, test

Для этого есть скрипт split_dataset.sh, в нем нужно указать следующие параметры
- dataset - ricechem / averitec / tabfact
- data-path - путь до директории с датасетом

По умолчанию датасет разбивается в соотношении 75/15/15 (но можно задать и свои пропорции). Дополнительные параметры можно посмотреть в скрипте split_datasets.py.

Пример split_dataset.sh:
```
python split_datasets.py \
    --dataset averitec \
    --data-path ../statics/datasets/AVeriTeC/data \
    --force
```

На выходе скрипт запишет train, val и test директории в папку --data-path.

## 2. Подготовка DPO данных

Для этого нужно запустить скрипт prepare_dpo_data.sh. В нем указываются следующие параметры:
- dataset - ricechem / averitec / tabfact
- data-path - путь до train данных
- output - .jsonl файл, куда записать DPO данные

Пример prepare_dpo_data.sh:
```
python prepare_dpo_data_intervention.py \
    --dataset averitec \
    --data-path ../statics/datasets/AVeriTeC/data/train \
    --output data/averitec_dpo_train.jsonl
```

Скрипт использует механизм интервенции, реализованный в make_intervention, и собирает данные с интервенциями для dpo

## 3. Дообучение модели с помощью DPO

Перед обучением нужно проверить, что установлены следующие библиотеки:
```
pip install torch transformers trl peft datasets accelerate numpy
```

Чтобы дообучить модель, нужно запустить скрипт run_dpo.sh, указав в нем следующие параметры:
- train-file - файл с dpo данными с предыдущего шага
- output-dir - директрия, в которую будут сохраняться чекпоинты
- save-steps - как часто сохранять чекпоинт
- faithfulness-dataset - датасет для оценки faithfulness (ricechem / averitec / tabfact)
- faithfulness-data-path - val датасет для замера faithfulness
- faithfulness-eval-batch-size - batch size для замера faithfulness на eval

Есть также много других параметров, которые относятся к настройкам обучения. Их можно посмотреть в ArgumentParser в скрипте dpo.py.

Замер faithfulness происходит каждый раз при сохранении чекпоинта, то есть раз в save-steps шагов.

```
python dpo.py \
    --train-file data/averitec_dpo_train.jsonl \
    --output-dir checkpoints/averitec_v1 \
    --save-steps 1 \
    --faithfulness-dataset averitec \
    --faithfulness-data-path ../statics/datasets/AVeriTeC/data/val \
    --faithfulness-eval-batch-size 32
```

При обучении настроено логирование в tensorboard, логи пишутся в {output_dir}/runs/{timestamp}_{hostname}/. Краткая инструкция для просмотра tensorboard:
1) pip install tensorboard
2) запустить обучение (run_dpo.sh)
3) запуск UI: 
```
tensorboard --logdir <output_dir>/runs --port
  6006
```
4) логи появятся в http://localhost:6006 (возможно придется пробросить порт с удаленной машины: ssh -L 6006:localhost:6006 user@host)

## 4. Просмотр метрик

Помимо tensorboard можно отдельно построить графики faithfulness и performance после DPO. Для этого можно воспользоваться скриптом plot_failthfulness.sh. Пример скрипта:
```
python plot_faithfulness_from_trainer_state.py \
    checkpoints/ricechem_v2 \
    --output plots/ricechem_v2.png
```
Первым аргументов нужно передать output_dir с предыдущей стадии. Скрипт возьмет последний чекпоинт и построит графики по истории метрик этого чекпоинта.