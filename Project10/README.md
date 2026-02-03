Project Summary
This project implements and compares recurrent neural networks (RNN, GRU, LSTM) for dual tasks of name generation and gender classification on a names dataset, analyzing vanishing gradients, multitask learning, perplexity metrics, and embedding visualizations.

📌 Contents

1. Introduction & RNN Fundamentals
- Описаны типы задач для RNN (sequence modeling: машинный перевод, генерация текста, распознавание речи).
- Приведены примеры архитектур: one-to-many (image captioning), many-to-many (машинный перевод), many-to-one (sentiment analysis).
- Нарисована схема обработки слова "Hello" языковой RNN с блоками embedding → RNN → linear + параметры hidden state (h₀ = zeros).
- Сравнение vanilla RNN vs LSTM/GRU (gates для борьбы с vanishing gradient) + схема LSTM/GRU.

2. Data Analysis & Preparation
- Построены гистограммы длин имен, проверен баланс по гендеру, посчитана частота букв.
- Кодирование пола, приведение к lowercase, добавление <SOS>/<EOS>, padding до max_length.
- Создан токенайзер (буквы + спецсимволы) с bidirectional mapping, преобразование имен в последовательности индексов.
- Разделение на train/valid/test (80-10-10%) с random_state.

3. Vanilla RNN Implementation
- Реализована сеть с нуля: Embedding → custom RNN → Linear(next token) + Linear(gender) + Sigmoid.
- Обучение на next-letter prediction (cross-entropy, teacher forcing), генерация стохастическая (softmax с temperature).
- Evaluation: графики loss, 10 сгенерированных имен, ROC AUC на test для gender classification.

4. GRU & LSTM Comparison
- Повторены шаги 4 для GRU и LSTM (замена recurrent block).
- Сравнение сходимости loss, качества генерации, ROC AUC по гендеру для всех 3 архитектур.

5. Perplexity Metric Analysis
- Вывод связи perplexity ↔ cross-entropy, 2 реализации (с/без CE loss).
- Рассчитаны baseline значения: random model и most-popular-letters model.

6. Gender Classification Training
- Переобучение RNN/GRU/LSTM только на gender loss (BCE vs NLL), объяснение выбора лучшей loss.
- Evaluation: loss curves, сгенерированные имена, ROC AUC на test.

7. Vanishing Gradient Analysis
- Подмножество данных с fixed-length именами (mode), сбор градиентов embedding layer (register_backward_hook).
- Отладка на batch_size=1, анализ Frobenius norm градиентов по позициям токенов.
- Сравнение распределений mean gradient norm (RNN vs GRU vs LSTM) для next-letter и gender задач.

8. Multitask Learning
- Обучение GRU/LSTM с joint loss (next-letter + gender classification).
- Evaluation: dual loss curves, сгенерированные имена, ROC AUC по гендеру.

9. Embeddings Visualization
- t-SNE и PCA снижение размерности embedding'ов букв из multitask модели.
- Scatterplot гласных vs согласных, анализ кластеризации в 2D пространстве.