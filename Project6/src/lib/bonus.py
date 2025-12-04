import os
import zipfile
import tempfile
import numpy as np
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import shutil
import matplotlib.pyplot as plt

class FaceExpressionCompressor:
    def __init__(self, happy_archive, sad_archive, happy_label='happy', sad_label='sad', n_components=100, batch_size=100):
        self.happy_archive = happy_archive
        self.sad_archive = sad_archive
        self.happy_label = happy_label
        self.sad_label = sad_label
        self.n_components = n_components
        self.batch_size = batch_size
        
        self.scaler = StandardScaler()
        self.ipca = IncrementalPCA(n_components=self.n_components)
        
        self.X_compressed = None
        self.y = None
        
        # Временная директория для распаковки архивов
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Буфер для хранения векторов при обучении
        self._buffer = []
        
    def _extract_all_images(self, archive_path, subfolder_label = None):
        # Создаёт отдельную временную папку для каждого архива
        if not subfolder_label:
            subfolder_label = archive_path.split("/")[-1].split("zip")[0]
        temp_subdir = os.path.join(self.temp_dir.name, subfolder_label)
        os.makedirs(temp_subdir, exist_ok=True)
        
        extracted_files = []
        with zipfile.ZipFile(archive_path, 'r') as z:
            z.extractall(temp_subdir)
            for root, _, files in os.walk(temp_subdir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        extracted_files.append(os.path.join(root, file))
        return extracted_files
    
    def _image_to_vector(self, image_path, target_size=(48, 48)):
        # Загружает и преобразует изображение в вектор
        try:
            with Image.open(image_path) as img:
                img_gray = img.convert('L').resize(target_size)
                vec = np.array(img_gray).flatten()
                return vec
        except Exception as e:
            print(f"Ошибка при загрузке {image_path}: {e}")
            return None
        
    def _process_archive(self, archive_path, label):
        # Извлекаем файлы
        image_files = self._extract_all_images(archive_path)
        vectors = []
        
        for img_path in image_files:
            vec = self._image_to_vector(img_path)
            if vec is not None:
                vectors.append(vec)
                
                # Накопление батча для обучения
                self._buffer.append(vec)
                if len(self._buffer) >= self.batch_size:
                    self._partial_fit_batch()
        
        # Обработать остатки
        if len(self._buffer) > 0:
            self._partial_fit_batch()
        
        return vectors
    
    def _partial_fit_batch(self):
        # Обучаем IncrementalPCA набатче
        batch = np.vstack(self._buffer)
        # Масштабируем
        batch_scaled = self.scaler.transform(batch) if hasattr(self.scaler, 'mean_') else self.scaler.fit_transform(batch)
        self.ipca.partial_fit(batch_scaled)
        # Очищаем буфер
        self._buffer.clear()
    
    def fit(self):
        # Основной метод обучения
        self._buffer.clear()
        self.ipca = IncrementalPCA(n_components=self.n_components)
        self.scaler = StandardScaler()
        
        print("Обрабатываем веселые лица...")
        self._process_archive(self.happy_archive, self.happy_label)
        print("Обрабатываем грустные лица...")
        self._process_archive(self.sad_archive, self.sad_label)
        
    def transform_all(self):
        # После обучения преобразует все изображения, возвращает X_compressed и y
        X_vecs = []
        y_labels = []
        
        print("Преобразуем веселые лица")
        happy_vecs = [self._image_to_vector(p) for p in self._extract_all_images(self.happy_archive)]
        happy_vecs = [v for v in happy_vecs if v is not None]
        X_vecs.extend(happy_vecs)
        y_labels.extend([self.happy_label]*len(happy_vecs))
        
        print("Преобразуем грустные лица")
        sad_vecs = [self._image_to_vector(p) for p in self._extract_all_images(self.sad_archive)]
        sad_vecs = [v for v in sad_vecs if v is not None]
        X_vecs.extend(sad_vecs)
        y_labels.extend([self.sad_label]*len(sad_vecs))
        
        X_vecs = np.vstack(X_vecs)
        X_scaled = self.scaler.transform(X_vecs)
        X_compressed = self.ipca.transform(X_scaled)
        self.X_compressed = X_compressed
        self.y = np.array(y_labels)
    
    def latent_arithmetic(self, src_label, dst_label):
        # Вычисляет сдвиг от src к dst (например, от sad к happy, или наоборот)
        src_vec = self.X_compressed[self.y == src_label].mean(axis=0)
        dst_vec = self.X_compressed[self.y == dst_label].mean(axis=0)
        if not np.allclose(src_vec, dst_vec): 
            print("Средние векторы различаются.")
        else:
            print("Средние векторы совпадают! Проверьте данные.")
        return dst_vec - src_vec  # направление смещения
    
    def transform_face(self, face_index=0, src_label=None, dst_label=None):
        # Позволяет выбрать исходную эмоцию и желаемую целевую эмоцию
        assert src_label is not None and dst_label is not None, "Передайте оба label"
        # Находим индексы исходной эмоции
        idx = np.where(self.y == src_label)[0][face_index]
        src_vec = self.X_compressed[idx]
        # Сдвиг
        diff = self.latent_arithmetic(src_label, dst_label)
        transformed_vec = src_vec + diff
        return src_vec, transformed_vec
    
    def plot_transformation(self, face_index=0, src_label=None, dst_label=None, target_size=(48,48)):
        """
        Отрисовывает:
        1. Cырой исходник из архива (src_label)
        2. Восстановленное сжатое с помощью PCA (src_label)
        3. Преобразованное (dst_label)
        """
        # Получаем список всех исходных файлов (например, sad или happy)
        paths = self._extract_all_images(
            self.sad_archive if src_label == self.sad_label else self.happy_archive,
            src_label
        )
        raw_image = None
        if face_index < len(paths):
            raw_image = Image.open(paths[face_index]).convert('L').resize(target_size)
            raw_image = np.array(raw_image)
        else:
            print(f"Индекс {face_index} вне диапазона, всего {len(paths)} изображений.")
            return
        
        # Получаем исходный и преобразованный вектора
        src_vec, transf_vec = self.transform_face(face_index, src_label, dst_label)
        if not np.allclose(src_vec, transf_vec):
            print("Векторы различаются.")
        else:
            print("Векторы совпадают.")
        
        # Восстанавливаем изображения
        src_recon_scaled = self.ipca.inverse_transform(src_vec.reshape(1, -1))
        transf_recon_scaled = self.ipca.inverse_transform(transf_vec.reshape(1, -1))
        src_recon = self.scaler.inverse_transform(src_recon_scaled)
        transf_recon = self.scaler.inverse_transform(transf_recon_scaled)
        
        src_recon_img = src_recon.reshape(target_size)
        transf_img = transf_recon.reshape(target_size)
        
        # Рисуем
        fig, axes = plt.subplots(1, 3, figsize=(8, 3))
        axes[0].imshow(raw_image, cmap='gray', interpolation='nearest')
        axes[0].set_title(f'Raw {src_label} face #{face_index}', fontsize=8)
        axes[0].axis('off')

        axes[1].imshow(src_recon_img, cmap='gray', interpolation='nearest')
        axes[1].set_title(f'PCA recon {src_label}', fontsize=8)
        axes[1].axis('off')

        axes[2].imshow(transf_img, cmap='gray', interpolation='nearest')
        axes[2].set_title(f'Transformed {dst_label}', fontsize=8)
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
    
    def cleanup(self):
        # Удаляет временную папку с распаковкой
        if self.temp_dir:
            self.temp_dir.cleanup()
            print("Временные файлы удалены")
