import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import zipfile
import tempfile
import shutil


def _find_and_extract_recursive(zip_file_obj, video_filename, temp_dir, current_depth=0):
    """
    (Вспомогательная функция) Рекурсивно ищет файл в открытом zip-объекте.
    """
    indent = "  " * current_depth
    print(f"{indent}- Поиск в архиве '{os.path.basename(zip_file_obj.filename)}'...")

    for item_info in zip_file_obj.infolist():
        # Если нашли целевое видео
        if item_info.filename.endswith(video_filename):
            print(f"{indent}  -> НАЙДЕН ЦЕЛЕВОЙ ФАЙЛ: '{item_info.filename}'")
            zip_file_obj.extract(item_info, path=temp_dir)
            return os.path.join(temp_dir, item_info.filename)
            
        # Если нашли вложенный архив
        elif item_info.filename.endswith('.zip'):
            print(f"{indent}  -> Найден вложенный архив: '{item_info.filename}'")
            
            # Извлекаем вложенный архив во временную папку
            nested_archive_path = os.path.join(temp_dir, item_info.filename)
            zip_file_obj.extract(item_info, path=temp_dir)
            
            # Рекурсивно вызываем самих себя для этого вложенного архива
            try:
                with zipfile.ZipFile(nested_archive_path, 'r') as nested_zip:
                    result_path = _find_and_extract_recursive(nested_zip, video_filename, temp_dir, current_depth + 1)
                    if result_path:
                        return result_path # Если нашли, сразу возвращаем результат наверх
            except zipfile.BadZipFile:
                print(f"{indent}  -> Ошибка: '{item_info.filename}' не является корректным ZIP-архивом.")
                continue

    print(f"{indent}- Файл не найден на этом уровне.")
    return None # Если ничего не нашли на этом уровне

def process_video_from_nested_archive(archive_path, video_filename):
    """
    Рекурсивно находит видеофайл внутри вложенных ZIP-архивов, 
    извлекает его и обрабатывает с помощью video_to_matrix.
    """
    if not os.path.exists(archive_path):
        print(f"Ошибка: Архив не найден по пути {archive_path}")
        return None, None

    # Используем временную директорию для всех операций
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Создана общая временная директория: {temp_dir}\n")
        
        try:
            with zipfile.ZipFile(archive_path, 'r') as main_zip:
                # Начинаем рекурсивный поиск
                extracted_file_path = _find_and_extract_recursive(main_zip, video_filename, temp_dir)
                
                if extracted_file_path:
                    print("\n--- Файл успешно извлечен, начинаем обработку видео ---")
                    return video_to_matrix(extracted_file_path)
                else:
                    print(f"\nИТОГ: Файл '{video_filename}' не был найден ни в одном из вложенных архивов.")
                    return None, None

        except zipfile.BadZipFile:
            print(f"Ошибка: Файл '{archive_path}' не является корректным ZIP-архивом.")
            return None, None
        except Exception as e:
            print(f"Произошла непредвиденная ошибка: {e}")
            return None, None
        
# --- 1. Функция загрузки видео и преобразования в матрицу ---

def video_to_matrix(video_path):
    """
    Загружает видео, преобразует каждый кадр в оттенки серого, 
    вытягивает в вектор и собирает в одну большую матрицу.
    
    Args:
        video_path (str): Путь к видеофайлу.
        
    Returns:
        tuple: (X, frame_shape)
            X (np.ndarray): Матрица, где каждая колонка - это один кадр.
            frame_shape (tuple): Форма исходного кадра (высота, ширина) 
                                 для последующего восстановления.
    """
    if not os.path.exists(video_path):
        print(f"Ошибка: Видеофайл не найден по пути {video_path}")
        return None, None
        
    cap = cv2.VideoCapture(video_path)
    
    frame_list = []
    
    # Получаем форму кадра с первого же успешного считывания
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось прочитать видео.")
        return None, None
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_shape = gray_frame.shape
    frame_list.append(gray_frame.flatten())
    
    # Читаем остальные кадры
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_list.append(gray_frame.flatten())
        
    cap.release()
    
    # Собираем матрицу X, где каждая колонка - это кадр
    # Используем тип float для SVD
    X = np.array(frame_list, dtype=float).T
    
    print(f"Видео успешно преобразовано в матрицу формы: {X.shape}")
    print(f"Исходная форма кадра: {frame_shape}")
    
    return X, frame_shape

# --- 2. Функция для анализа SVD и выбора ранга ---

def plot_svd_spectrum_for_video(s):
    """
    Строит два графика для анализа сингулярных значений видео:
    1. Спектр сингулярных значений (показывает их величину).
    2. График накопленной дисперсии (показывает вклад компонент).
    
    Args:
        s (np.ndarray): 1D-массив сингулярных значений из SVD.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # График 1: Спектр сингулярных значений
    axes[0].plot(range(1, len(s) + 1), s, 'b-')
    axes[0].set_yscale('log')
    axes[0].set_title("Спектр сингулярных значений (log-шкала)")
    axes[0].set_xlabel("Ранг (компонента)")
    axes[0].set_ylabel("Величина сингулярного значения")
    axes[0].grid(True, which="both", ls="--")
    # Вертикальная линия для акцента на первых компонентах
    axes[0].axvline(x=5, linestyle='--', color='red', label='Первые 5 компонент')
    axes[0].legend()
    
    # График 2: Накопленная объясненная дисперсия
    variance_explained = np.cumsum(s**2) / np.sum(s**2)
    axes[1].plot(range(1, len(s) + 1), variance_explained, 'r-')
    axes[1].set_title("Накопленная объясненная дисперсия")
    axes[1].set_xlabel("Ранг (компонента)")
    axes[1].set_ylabel("Доля общей дисперсии")
    axes[1].grid(True)
    axes[1].set_ylim(0, 1.05)
    
    plt.suptitle("Анализ компонент SVD для выбора ранга фона", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- 3. Функция для восстановления кадра по рангу ---

def reconstruct_frame_from_svd(U, s, Vt, rank, frame_index, original_shape):
    """
    Восстанавливает видеоматрицу с заданным рангом и возвращает 
    один конкретный кадр из нее.
    
    Args:
        U, s, Vt: Компоненты SVD разложения видеоматрицы.
        rank (int): Ранг для низкоранговой аппроксимации.
        frame_index (int): Индекс кадра, который нужно восстановить (например, 0 для первого).
        original_shape (tuple): Форма кадра (высота, ширина) для reshape.
        
    Returns:
        np.ndarray: 2D-массив, представляющий восстановленный кадр.
    """    
    # Низкоранговое приближение всей видеоматрицы
    reconstructed_video_matrix = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
    
    # Извлекаем нужную колонку (кадр)
    reconstructed_frame_vector = reconstructed_video_matrix[:, frame_index]
    
    # Преобразуем вектор обратно в 2D-матрицу изображения
    reconstructed_frame = reconstructed_frame_vector.reshape(original_shape)
    
    return reconstructed_frame

def plot_reconstruction(original_first_frame, ranks_to_reconstruct, U, s, Vt, frame_shape):
    num_ranks = len(ranks_to_reconstruct)
    fig, axes = plt.subplots(1, num_ranks + 1, figsize=(5 * (num_ranks + 1), 5))
    
    axes[0].imshow(original_first_frame, cmap='gray')
    axes[0].set_title("Оригинальный 1-й кадр")
    axes[0].axis('off')
    
    for i, rank in enumerate(ranks_to_reconstruct):
        # 4. Восстановление первого кадра (frame_index=0) с разным рангом
        background_frame = reconstruct_frame_from_svd(U, s, Vt, rank, 0, frame_shape)
        
        axes[i+1].imshow(background_frame, cmap='gray')
        axes[i+1].set_title(f"Фон (ранг = {rank})")
        axes[i+1].axis('off')
        
    plt.suptitle("Сравнение оригинального кадра с выделенным фоном", fontsize=16)
    plt.show()