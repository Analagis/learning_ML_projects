import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import warnings
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter

from models import RecurrentModel, VanillaRNNBlock, GRUBlock, LSTMBlock
from nameTrainer import NameTrainer
from utils import build_vocab, compute_roc_auc_splits, calculate_perplexity

class ModelRunner:
    def __init__(
        self,
        rnn_block_classes: List[str],  # ['VanillaRNNBlock', 'GRUBlock', 'LSTMBlock']
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        dropout_p: float = 0,
        lr: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.rnn_block_classes = rnn_block_classes
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.lr = lr
        self.device = device
        
        # словари результатов
        self.models_name: Dict[str, RecurrentModel] = {}
        self.models_gender: Dict[str, RecurrentModel] = {}
        self.histories: Dict[str, Dict] = {}
        self.roc_aucs: Dict[str, Dict] = {}
        self.generated_names: Optional[pd.DataFrame] = None
        
        # блок-классы
        self.block_map = {
            'VanillaRNNBlock': VanillaRNNBlock,
            'GRUBlock': GRUBlock,
            'LSTMBlock': LSTMBlock,
        }
        
        print(f"ModelRunner инициализирован для блоков: {rnn_block_classes}")

    def train_models(
        self,
        train_loader,
        valid_loader,
        num_epochs: int,
        use_next_char_loss: bool = False,
        use_gender_loss: bool = False,
        alpha: float = 1.0,
        beta: float = 1.0
    ):
        """Обучает все модели и сохраняет history."""
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs
        
        for block_name in self.rnn_block_classes:
            print(f"\n{'='*50}")
            print(f"Обучение {block_name}...")
            print(f"{'='*50}")
            
            # создаём модель
            rnn_block_cls = self.block_map[block_name]
            model = RecurrentModel(
                vocab_size=self.vocab_size,
                embedding_dim=self.embedding_dim,
                hidden_size=self.hidden_size,
                rnn_block_cls=rnn_block_cls,
                dropout_p=self.dropout_p
            )
            
            trainer = NameTrainer(
                model,
                use_next_char_loss=use_next_char_loss,
                use_gender_loss=use_gender_loss,
                alpha=alpha, beta=beta
            )
            
            # обучаем
            history = trainer.fit(train_loader, valid_loader, num_epochs)

            if use_next_char_loss:
                self.models_name[block_name] = model
            if use_gender_loss:
                self.models_gender[block_name] = model

            self.histories[block_name] = history
            
            print(f"{block_name}: обучение завершено")
    
    def plot_losses(self, loss_type: str = 'next_char', figsize=(12, 8)):
        """
        Отрисовка сходимости loss для всех моделей.
        
        loss_type: 'next_char' | 'gender' | 'both' | 'total'
        """
        plt.figure(figsize=figsize)
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.histories)))
        
        for i, (block_name, history) in enumerate(self.histories.items()):
            color = colors[i]
            
            if loss_type == 'next_char' or loss_type == 'both':
                # next_char loss
                train_next = np.array([v for v in history['train_next'] if v is not None])
                valid_next = np.array([v for v in history['valid_next'] if v is not None])
                
                if len(train_next) > 0:
                    plt.plot(train_next, label=f'{block_name}_train_next', 
                            color=color, linestyle='-', alpha=0.8, linewidth=2)
                if len(valid_next) > 0:
                    plt.plot(valid_next, label=f'{block_name}_valid_next', 
                            color=color, linestyle='--', alpha=0.8, linewidth=2)
            
            if loss_type == 'gender' or loss_type == 'both':
                # gender loss
                train_gender = np.array([v for v in history['train_gender'] if v is not None])
                valid_gender = np.array([v for v in history['valid_gender'] if v is not None])
                
                if len(train_gender) > 0:
                    plt.plot(train_gender, label=f'{block_name}_train_gender', 
                            color=color, linestyle='-', alpha=0.7, linewidth=1.5)
                if len(valid_gender) > 0:
                    plt.plot(valid_gender, label=f'{block_name}_valid_gender', 
                            color=color, linestyle='--', alpha=0.7, linewidth=1.5)
            
            if loss_type == 'total':
                # общий loss
                train_total = np.array(history['train_total'])
                valid_total = np.array(history['valid_total'])
                
                plt.plot(train_total, label=f'{block_name}_train_total', 
                        color=color, linestyle='-', alpha=0.8)
                plt.plot(valid_total, label=f'{block_name}_valid_total', 
                        color=color, linestyle='--', alpha=0.8)
        
        plt.xlabel('Эпоха')
        plt.ylabel('Loss')
        plt.title(f'Сходимость loss ({loss_type})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_names_comparison(
        self,
        temperatures: List[float] = [0.7, 1.0, 1.3],
        n_samples: int = 5,
        max_len: int = 13,
        token2id: None|Dict = None,
        id2token: None|Dict = None
    ):
        """Генерация имён всеми моделями."""
        if token2id is None or id2token is None:
            raise ValueError("Нужны token2id и id2token для генерации")
            
        data_by_temp = {f'temp_{t}': [] for t in temperatures}
        model_names_index = [] # Для индекса DataFrame

        if not self.models_name:
            models = self.models_gender
        else:
            models = self.models_name

        for block_name, model in models.items():
            model_names_index.append(block_name)
            
            # Проходим по всем температурам для текущей модели
            for temp in temperatures:
                # Генерируем пачку имен (n_samples штук) для этой температуры
                raw_samples = [
                    model.generate_name(
                        token2id, id2token, 
                        start_text="", max_len=max_len, temperature=temp
                    ) for _ in range(n_samples)
                ]
                
                clean_string = ", ".join(raw_samples)

                # Добавляем список имен в соответствующий столбец
                data_by_temp[f'temp_{temp}'].append(clean_string)
        
        self.generated_names = pd.DataFrame(data_by_temp, index=model_names_index)
        
        print("\nСгенерированные имена:")
        display(self.generated_names)

    def compute_roc_auc(self, test_loader):
        """ROC AUC для всех моделей."""
        self.test_loader = test_loader
        roc_results = {}
        
        if not self.models_gender:
            models = self.models_name
        else:
            models = self.models_gender

        for block_name, model in models.items():
            roc = compute_roc_auc_splits(
                model, self.train_loader, self.valid_loader, self.test_loader, self.device
            )
            roc_results[block_name] = roc

        self.roc_aucs = roc_results
        
        # DataFrame для удобства
        auc_df = pd.DataFrame(roc_results).T
        auc_df.columns = ['train_roc_auc', 'valid_roc_auc', 'test_roc_auc']
        print("\nROC AUC по моделям:")
        display(auc_df.round(3))

    def compute_perplexity(self):
        """Считает Perplexity для всех моделей на тестовой выборке."""
        ppl_results = {}
        
        # Выбираем модели (гендерные или обычные)
        models = self.models_gender if self.models_gender else self.models_name
        
        for block_name, model in models.items():
            model.eval()
            all_logits = []
            all_targets = []
            
            # Прогон по тестовому лоадеру
            with torch.no_grad():
                for x_batch, y_batch in self.test_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # 1. Получаем выход модели
                    output = model(x_batch)
                    
                    # Обработка разных типов выхода (кортеж или тензор)
                    if isinstance(output, tuple):
                        logits = output[0] 
                    else:
                        logits = output

                    # 2. Сохраняем логиты и таргеты
                    # Переносим на CPU, чтобы не забить память GPU
                    all_logits.append(logits[:, :-1, :].cpu())
                    all_targets.append(x_batch[:, 1:].cpu())
            
            # Объединяем все батчи в один большой тензор
            if not all_logits:
                print(f"⚠️ Лоадер пуст для модели {block_name}")
                continue
                
            full_logits = torch.cat(all_logits, dim=0)   # (Total_N, Seq_Len, Vocab)
            full_targets = torch.cat(all_targets, dim=0) # (Total_N, Seq_Len)
            
            # 3. Считаем метрики, используя нашу функцию calculate_perplexity
            # Предполагаем, что она определена глобально или как статический метод
            # Если она внутри класса, добавьте self. перед вызовом

            ppl_metrics = calculate_perplexity(full_logits, full_targets, ignore_index=0) # 0 обычно паддинг
            
            ppl_results[block_name] = ppl_metrics

        # Сохраняем и выводим результаты
        self.ppl_results = ppl_results
        
        # Формируем красивую таблицу
        ppl_df = pd.DataFrame(ppl_results).T
        # Округляем для красоты
        ppl_df = ppl_df.round(4)
        
        print("\nPerplexity по моделям:")
        display(ppl_df)

    def visualize_embeddings(self, id2token, model_name_key=None):
        """
        Визуализирует эмбеддинги букв с помощью PCA и t-SNE.
        
        Args:
            model_name_key (str): Ключ модели в словаре self.models_gender. 
                                Если None, берет первую попавшуюся модель.
        """
        
        # 1. Получаем модель (предполагаем, что это Multitask модель с эмбеддингами)
        if self.models_gender is None or len(self.models_gender) == 0:
            print("Нет доступных моделей с эмбеддингами (models_gender пуст).")
            return

        if model_name_key is None:
            model_name_key = list(self.models_gender.keys())[0]
        
        model = self.models_gender[model_name_key]
        print(f"Используем модель: {model_name_key}")

        # 2. Извлекаем матрицу эмбеддингов
        # Обычно это model.embedding.weight или model.emb.weight
        # Проверяем атрибуты, так как имя слоя может отличаться
        if hasattr(model, 'embedding'):
            emb_matrix = model.embedding.weight.detach().cpu().numpy()
        elif hasattr(model, 'emb'):
            emb_matrix = model.emb.weight.detach().cpu().numpy()
        else:
            # Пытаемся найти слой nn.Embedding рекурсивно
            emb_layer = None
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    emb_layer = module
                    break
            
            if emb_layer is None:
                print("Не удалось найти слой Embeddings в модели.")
                return
            emb_matrix = emb_layer.weight.detach().cpu().numpy()

        # Убираем паддинг (обычно индекс 0), если он есть и не несет смысла
        # Но для чистоты картинки можно оставить, или убрать вручную
        # emb_matrix = emb_matrix[1:] 
        
        n_tokens, _ = emb_matrix.shape

        # 3. Подготовка данных: Гласные и Согласные
        vowels_str = "aeiouy" # Можно расширить
        labels = []
        colors = []
        tokens = []

        for i in range(n_tokens):
            char = id2token.get(i, str(i))

            tokens.append(char)
            
            # Определяем тип (Гласная / Согласная / Спецсимвол)
            char_lower = char.lower()
            if char_lower in vowels_str:
                labels.append("Vowel")
                colors.append("red")
            elif char_lower.isalpha():
                labels.append("Consonant")
                colors.append("blue")
            else:
                labels.append("Special") # <PAD>, <EOS>, <BOS>
                colors.append("gray")

        # 4. Сжатие размерности
        
        # PCA
        print("Обучаем PCA...")
        pca = PCA(n_components=2, random_state=42)
        emb_pca = pca.fit_transform(emb_matrix)
        
        # t-SNE
        print("Обучаем t-SNE...")
        # perplexity должна быть < n_samples. Обычно 30, но для алфавита (26-30 букв) лучше взять меньше, например 5-10
        perp = min(30, n_tokens - 1) 
        tsne = TSNE(n_components=2, perplexity=6, random_state=42, init='pca', learning_rate='auto')
        emb_tsne = tsne.fit_transform(emb_matrix)

        # 5. Визуализация (два графика рядом)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Функция для рисования
        def plot_scatter(ax, data, title):
            # Рисуем точки
            # Используем seaborn для удобной легенды по цветам, или matplotlib напрямую
            unique_labels = list(set(labels))
            
            for label_type in unique_labels:
                indices = [i for i, l in enumerate(labels) if l == label_type]
                if not indices: continue
                
                x = data[indices, 0]
                y = data[indices, 1]
                c = colors[indices[0]] # берем цвет первого элемента этой группы
                
                ax.scatter(x, y, c=c, label=label_type, alpha=0.7, s=100, edgecolors='k')
                
            # Подписываем буквы
            for i, txt in enumerate(tokens):
                # Можно не подписывать спецсимволы, чтобы не засорять
                if labels[i] != "Special": 
                    ax.annotate(txt, (data[i, 0], data[i, 1]), xytext=(5, 2), textcoords='offset points', fontsize=12, fontweight='bold')
            
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Строим графики
        plot_scatter(axes[0], emb_tsne, "t-SNE Projection")
        plot_scatter(axes[1], emb_pca, "PCA Projection")
        
        plt.tight_layout()
        plt.show()

    def analyze_vanishing_gradient(self, X_data, y_data, token2id, task_type='gender', n_epochs=3, batch_size=32):
        print(f"\n=== Запуск анализа Vanishing Gradient (Task: {task_type}) ===")
        
        # Превращаем в тензоры
        if not torch.is_tensor(X_data):
            X_data = torch.tensor(X_data, dtype=torch.long)
        if not torch.is_tensor(y_data):
            # Для gender y должен быть float (0.0/1.0) для BCEWithLogitsLoss
            vals = y_data.values if hasattr(y_data, 'values') else y_data
            y_data = torch.tensor(vals).float() # Явно float!

        # --- Шаг A: Фильтрация данных по Моде ---
        pad_idx = token2id.get('<PAD>', 0)
        bos_idx = token2id.get('<', -1)
        eos_idx = token2id.get('>', -1)
        
        mask = (X_data != pad_idx)
        if bos_idx != -1: mask &= (X_data != bos_idx)
        if eos_idx != -1: mask &= (X_data != eos_idx)
        lengths = mask.sum(dim=1).numpy()
        
        mode_len = int(Counter(lengths).most_common(1)[0][0])
        print(f"Самая популярная длина (Mode): {mode_len}")
        
        indices = np.where(lengths == mode_len)[0]
        subset_dataset = TensorDataset(X_data[indices], y_data[indices])
        loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        results = {} 
        models_to_test = self.models_gender if task_type == 'gender' else self.models_name
        
        for model_name, model in models_to_test.items():
            print(f"Анализ: {model_name}")
            model.train()
            model.zero_grad()
            
            emb_layer = model.embedding # У вас точно есть self.embedding в RecurrentModel
            
            grads_storage = [] 

            # Хук на тензор (через forward hook слоя)
            def forward_hook_fn(module, inp, out):
                if out.requires_grad:
                    out.register_hook(lambda g: grads_storage.append(g.detach().cpu()))

            h = emb_layer.register_forward_hook(forward_hook_fn)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Лоссы
            criterion_gender = torch.nn.BCEWithLogitsLoss()
            criterion_name = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
            
            max_steps = 30
            
            for epoch in range(n_epochs):
                for step, (bx, by) in enumerate(loader):
                    if step >= max_steps: break
                    
                    bx = bx.to(self.device)
                    by = by.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # ВАШ RETURN: token_logits, gender_logit, gender_prob, last_hidden
                    
                    if task_type == 'gender':
                        # Для задачи пола подаем все имя целиком
                        # RecurrentModel.forward возвращает 4 элемента
                        # Нам нужен 2-й элемент: gender_logit (batch, 1)
                        _, gender_logit, _, _ = model(bx)
                        
                        # by должен быть (batch,), превращаем в (batch, 1) или наоборот
                        loss = criterion_gender(gender_logit.view(-1), by.view(-1))
                            
                    else: # 'name'
                        # Next token prediction: вход 0..N-1, цель 1..N
                        inp = bx[:, :-1]
                        tgt = bx[:, 1:]
                        
                        # Нам нужен 1-й элемент: token_logits (batch, seq, vocab)
                        token_logits, _, _, _ = model(inp)
                        
                        # CrossEntropy ожидает (N, C) и (N)
                        loss = criterion_name(
                            token_logits.reshape(-1, token_logits.size(-1)), 
                            tgt.reshape(-1)
                        )

                    loss.backward()
                    optimizer.step()
            
            h.remove()

            if grads_storage:
                full_grads = torch.cat(grads_storage, dim=0) # (Total_N, Seq, Emb)
                norms = torch.norm(full_grads, p='fro', dim=-1) # (Total_N, Seq)
                mean_norms = norms.mean(dim=0).numpy() # (Seq,)
                results[model_name] = mean_norms
            else:
                print("  [Warn] Градиенты не пойманы.")

        # --- Рисуем ---
        plt.figure(figsize=(10, 6))
        for name, values in results.items():
            plt.plot(values, marker='o', label=name, linewidth=2)
            
        plt.title(f"Gradient Norm Analysis ({task_type})")
        plt.xlabel("Token Position (Time Step)")
        plt.ylabel("Avg Gradient Norm")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

