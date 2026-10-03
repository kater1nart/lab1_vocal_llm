# Анализ ошибок: gemma3_4b_fewshot vs LogReg (full_test, n=1009)

## Распределение предсказаний LLM по истинным классам

| y_true       |   female |   group |   instrumental |   male |   All |
|:-------------|---------:|--------:|---------------:|-------:|------:|
| female       |       95 |      27 |              0 |    132 |   254 |
| group        |      112 |      12 |              0 |     20 |   144 |
| instrumental |      176 |      86 |              1 |     87 |   350 |
| male         |       89 |       6 |              0 |    166 |   261 |
| All          |      472 |     131 |              1 |    405 |  1009 |

## Уверенность LLM

| ok_llm   |   count |   mean |   min |   max |
|:---------|--------:|-------:|------:|------:|
| False    |     735 |   0.95 |  0.95 |  0.95 |
| True     |     274 |   0.95 |  0.95 |  0.95 |

## Accuracy в жанровых подгруппах

| подгруппа    |   n |   acc_llm |   acc_baseline |
|:-------------|----:|----------:|---------------:|
| rock=1       | 208 |     0.5   |          0.654 |
| rock=0       | 801 |     0.212 |          0.705 |
| opera=1      | 106 |     0.425 |          0.84  |
| opera=0      | 903 |     0.254 |          0.678 |
| classical=1  | 153 |     0.163 |          0.81  |
| classical=0  | 856 |     0.291 |          0.674 |
| electronic=1 |  77 |     0.117 |          0.506 |
| electronic=0 | 932 |     0.284 |          0.71  |
| pop=1        | 133 |     0.391 |          0.639 |
| pop=0        | 876 |     0.253 |          0.703 |

## Согласие LLM и baseline

| ok_llm     |   baseline верно |   baseline ошибка |
|:-----------|-----------------:|------------------:|
| LLM верно  |              215 |                59 |
| LLM ошибка |              486 |               249 |

## 10 характерных ошибок LLM

|   clip_id | artist                                      | title                      | y_true       | y_pred   |   confidence | y_base       |
|----------:|:--------------------------------------------|:---------------------------|:-------------|:---------|-------------:|:-------------|
|     43991 | Solace                                      | Saalik 2                   | female       | group    |         0.95 | instrumental |
|     32157 | Norine Braun                                | Little Lamb                | female       | group    |         0.95 | male         |
|     47766 | Norine Braun                                | Climb the Mountain         | female       | male     |         0.95 | female       |
|     35118 | Norine Braun                                | Cruel Streak               | female       | male     |         0.95 | male         |
|      6267 | St. Eliyah Childrens Choir                  | Blessed is the man         | group        | female   |         0.95 | group        |
|     14176 | Kiev Theological Academy and Seminary Choir | O gladsome Radiance        | group        | female   |         0.95 | male         |
|     36437 | Arthur Yoria                                | My Best Routines           | group        | male     |         0.95 | male         |
|     32160 | Norine Braun                                | Little Lamb                | group        | male     |         0.95 | male         |
|     28690 | Tilopa                                      | Roku                       | instrumental | female   |         0.95 | instrumental |
|     52579 | Jacob Heringman                             | De Navarez _ Mille regretz | instrumental | female   |         0.95 | instrumental |
